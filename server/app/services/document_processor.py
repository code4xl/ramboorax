from app.helpers.processor import extract_text_from_url, chunk_text_parallel_enhanced, chunk_text_smart
from app.helpers.embedder import embed_chunks_parallel
from app.helpers.retriever import get_similar_contexts
from app.helpers.llm_reasoner import generate_batch_answer
from app.helpers.cache_manager import load_vector_store_if_exists, save_vector_store
from app.helpers.cache_manager import generate_qa_cache_key, load_qa_cache_if_exists, save_qa_cache
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class DocumentProcessorService:
    async def process_document_and_questions(self, document_url: str, questions: list) -> list:
        start=time.time()
        isCachingEnabled = True
        print(f"🚀 DEBUG: Validated request received")
        print(f"📄 Document URL: {document_url}")
        print(f"❓ Questions: {questions}")
        print(f"❓ Questions count: {len(questions) if hasattr(questions, '__len__') else 'N/A'}")
        print(f"❓ First few questions: {questions[:2] if len(questions) > 0 else 'None'}")

        # NEW: Extract document first to determine processing strategy
        extraction_result = extract_text_from_url(document_url)
        
        if extraction_result["isError"]:
            print(f"❌ ERROR: {extraction_result['message']}")
            error_message = f"Unable to process document: {extraction_result['message']}"
            return [error_message] * len(questions)
        
        raw_text = extraction_result["text"]
        tables = extraction_result.get("tables", [])
        
        # Check if extracted text is meaningful
        if not raw_text or len(raw_text.strip()) < 50:
            error_message = "The document appears to be empty or contains insufficient content for processing."
            print(f"❌ ERROR: {error_message}")
            return [error_message] * len(questions)
        
        # NEW: Determine page count and processing strategy
        # page_count = self._estimate_page_count(raw_text, tables)
        # print(f"📄 Estimated pages: {page_count}")
        
        if len(raw_text) < 30000:
            print("📝 Small document detected - using direct processing without embeddings")
            return await self._process_small_document(raw_text, tables, questions)
        
        # EXISTING FLOW: For large documents, continue with vector store logic
        print("📚 Large document detected - using embedding-based processing")
        
        # Use the URL directly to load/save cache
        db = load_vector_store_if_exists(document_url)

        if db is not None:
            print("✅ Cached vector store enabled.")
        else:
            print("📥 Embedding new large document.")
            try:                        
                # chunks = chunk_text(raw_text)
                if extraction_result.get("file_type") == "xlsx":
                    chunks = chunk_text_smart(raw_text, extraction_result)
                else:
                    chunks = chunk_text_parallel_enhanced(raw_text, chunk_size=300, overlap=50, num_threads=4)
                # db = embed_chunks(chunks)
                db = embed_chunks_parallel(chunks, batch_size=50, num_threads=4, use_enhanced=True)
                save_vector_store(db, document_url)
                print("✅ Document processed and cached successfully.")
            except Exception as e:
                error_message = f"Failed to process document content: {str(e)}"
                print(f"❌ ERROR: {error_message}")
                return [error_message] * len(questions)

        if isCachingEnabled:
            # Check Q&A cache
            qa_cache_key = generate_qa_cache_key(document_url, questions)
            cached_qa = load_qa_cache_if_exists(qa_cache_key)
        
            if cached_qa:
                print("✅ Using cached Q&A answers.")
                # Map questions to answers maintaining input order
                cached_questions = cached_qa["questions"]
                cached_answers = cached_qa["answers"]
                
                # Create question-to-answer mapping
                qa_mapping = dict(zip(cached_questions, cached_answers))
                
                # Return answers in the same order as input questions
                answers = [qa_mapping.get(q, "❌ Answer not found") for q in questions]

            else:                 
                # NEW: Process with table awareness
                answers = await self._process_questions_with_tables(db, questions, tables)
                # Save to Q&A cache
                save_qa_cache(qa_cache_key, questions, answers)
        else:
            print("❌ Caching is disabled, processing questions normally.")
            # NEW: Process with table awareness
            answers = await self._process_questions_with_tables(db, questions, tables)
            
        stop=time.time()
        print(f"🕒 Total Time: {stop - start:.2f} seconds")
        return answers
    

    def _estimate_page_count(self, text: str, tables: list) -> int:
        """Estimate page count based on content"""
        # Count explicit page markers in tables
        page_numbers = set()
        for table in tables:
            page_numbers.add(table.get("page", 1))
        
        if page_numbers:
            return max(page_numbers)
        
        # Fallback: estimate based on text length (rough estimate)
        chars_per_page = 2000  # Approximate
        estimated = max(1, len(text) // chars_per_page)
        return estimated

    async def _process_small_document(self, text: str, tables: list, questions: list) -> list:
        """Process small documents without embeddings"""
        print("🔄 Processing small document directly")
        
        # Create contexts for each question by checking relevance
        contexts = []
        for question in questions:
            context_parts = [text]  # Always include main text
            
            # Check if question relates to any table content
            relevant_tables = self._find_relevant_tables(question, tables)
            if relevant_tables:
                print(f"📊 Found {len(relevant_tables)} relevant tables for question")
                for table in relevant_tables:
                    context_parts.append(f"\nTable from page {table['page']}:\n{table['content']}")
            
            combined_context = "\n\n".join(context_parts)
            # Convert to Document objects for consistency with existing code
            from langchain.docstore.document import Document
            contexts.append([Document(page_content=combined_context)])
        
        # Use existing batch answer generation
        answers = await generate_batch_answer(contexts, questions)
        return answers

    def _find_relevant_tables(self, question: str, tables: list) -> list:
        """Find tables that might contain relevant information for the question"""
        # relevant_tables = []
        # question_lower = question.lower()
        
        # # Extract key terms from question
        # key_terms = self._extract_question_keywords(question_lower)
        
        # for table in tables:
        #     table_content_lower = table['content'].lower()
            
        #     # Check if any key terms appear in table
        #     if any(term in table_content_lower for term in key_terms):
        #         relevant_tables.append(table)
        
        return tables

    def _extract_question_keywords(self, question: str) -> list:
        """Extract important keywords from question"""
        # Remove common words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'a', 'an', 'to', 'for', 'of', 'as', 'by', 'what', 'who', 'where', 'when', 'how', 'why'}
        
        # Split and filter
        words = question.split()
        keywords = []
        
        for word in words:
            clean_word = word.strip('.,?!()[]{}";:').lower()
            if len(clean_word) > 2 and clean_word not in stop_words:
                keywords.append(clean_word)
        
        return keywords

    async def _process_questions_with_tables(self, db, questions: list, tables: list) -> list:
        """Process questions with table awareness for large documents"""
        contexts = []
        
        for question in questions:
            # Get vector-based context
            from app.helpers.retriever import get_similar_contexts
            vector_context = get_similar_contexts(db, question, k=10)
            
            # Find relevant tables
            relevant_tables = self._find_relevant_tables(question, tables)
            
            # Combine contexts
            combined_docs = vector_context.copy()
            
            # Add table context if relevant
            if relevant_tables:
                print(f"📊 Adding {len(relevant_tables)} tables to context for question")
                from langchain.docstore.document import Document
                for table in relevant_tables:
                    table_doc = Document(
                        page_content=f"Table from page {table['page']}:\n{table['content']}"
                    )
                    combined_docs.append(table_doc)
            
            contexts.append(combined_docs)
        
        # Generate answers
        from app.helpers.llm_reasoner import generate_batch_answer
        answers = generate_batch_answer(contexts, questions)
        return answers

    async def _process_batches_parallel(self, db, question_batches, max_workers):
        """Process question batches in parallel for maximum efficiency"""
        
        def process_single_batch(batch_info):
            """Process a single batch of questions"""
            batch_idx, question_batch = batch_info
            batch_start = time.time()
            
            try:
                # Step 1: Parallel context retrieval for all questions in batch
                with ThreadPoolExecutor(max_workers=len(question_batch)) as context_executor:
                    context_futures = {
                        context_executor.submit(get_similar_contexts, db, q, k=5): q 
                        for q in question_batch
                    }
                    
                    contexts = []
                    for future in as_completed(context_futures):
                        try:
                            context = future.result()
                            contexts.append(context)
                        except Exception as e:
                            print(f"❌ Context retrieval error: {e}")
                            contexts.append([])  # Empty context as fallback
                
                # Step 2: Generate answers for the batch
                batch_answers = generate_batch_answer(contexts, question_batch)
                
                batch_time = time.time() - batch_start
                # print(f"✅ Batch {batch_idx + 1} completed in {batch_time:.2f}s ({len(question_batch)} questions)")
                
                return batch_idx, batch_answers
                
            except Exception as e:
                print(f"❌ Batch {batch_idx + 1} error: {e}")
                return batch_idx, ["Error processing question" for _ in question_batch]

        # Process all batches in parallel
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch processing tasks
            batch_futures = {
                loop.run_in_executor(executor, process_single_batch, (idx, batch)): idx
                for idx, batch in enumerate(question_batches)
            }
            
            # Collect results maintaining order
            batch_results = {}
            for future in asyncio.as_completed(batch_futures):
                try:
                    batch_idx, batch_answers = await future
                    batch_results[batch_idx] = batch_answers
                except Exception as e:
                    batch_idx = batch_futures[future]
                    print(f"❌ Batch {batch_idx + 1} failed: {e}")
                    batch_results[batch_idx] = ["Error" for _ in question_batches[batch_idx]]
        
        # Combine results in correct order
        all_answers = []
        for i in range(len(question_batches)):
            all_answers.extend(batch_results.get(i, []))
        
        return all_answers

    async def _process_batch_optimized(self, db, question_batch):
        """Optimized processing for a single batch with parallel context retrieval"""
        
        # Parallel context retrieval
        async def get_context_async(question):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, get_similar_contexts, db, question)
        
        # Get contexts for all questions in parallel
        context_tasks = [get_context_async(q) for q in question_batch]
        contexts = await asyncio.gather(*context_tasks, return_exceptions=True)
        
        # Handle any exceptions in context retrieval
        clean_contexts = []
        for i, context in enumerate(contexts):
            if isinstance(context, Exception):
                print(f"❌ Context error for question {i}: {context}")
                clean_contexts.append([])  # Empty fallback
            else:
                clean_contexts.append(context)
        
        # Generate answers for the batch
        return generate_batch_answer(clean_contexts, question_batch)

    async def _process_questions_normally(self, db, questions):
        # OPTIMIZED PARALLEL BATCH PROCESSING
        batch_size = 5
        max_workers = min(4, (len(questions) + batch_size - 1) // batch_size)  # Dynamic worker count
            
        print(f"🔄 Processing {len(questions)} questions in parallel with {max_workers} workers")
            
        # Create batches
        question_batches = [
            questions[i:i + batch_size] 
            for i in range(0, len(questions), batch_size)
        ]
            
        # Process batches in parallel
        return await self._process_batches_parallel(db, question_batches, max_workers)