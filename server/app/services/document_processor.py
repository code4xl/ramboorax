from app.helpers.processor import extract_text_from_url, chunk_text_parallel, chunk_text
from app.helpers.embedder import embed_chunks_parallel, embed_chunks
from app.helpers.retriever import get_similar_contexts, get_fallback_contexts
from app.helpers.llm_reasoner import generate_batch_answer
from app.helpers.cache_manager import load_vector_store_if_exists, save_vector_store
from app.helpers.cache_manager import generate_qa_cache_key, load_qa_cache_if_exists, save_qa_cache, calculate_realistic_delay
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class DocumentProcessorService:
    async def process_document_and_questions(self, document_url: str, questions: list) -> list:
        start=time.time()
        isCachingEnabled = False
        print(f"🚀 DEBUG: Validated request received")
        print(f"📄 Document URL: {document_url}")
        print(f"❓ Questions: {questions}")
        print(f"❓ Questions count: {len(questions) if hasattr(questions, '__len__') else 'N/A'}")
        print(f"❓ First few questions: {questions[:2] if len(questions) > 0 else 'None'}")

        # Use the URL directly to load/save cache
        db = load_vector_store_if_exists(document_url)

        if db is not None:
            print("✅ Using cached vector store.")
        else:
            print("📥 Downloading and embedding new document.")
            #Extract text with Error handling
            extraction_result = extract_text_from_url(document_url)
        
            if extraction_result["isError"]:
                print(f"❌ ERROR: {extraction_result['message']}")
                # Return the same error message for all questions
                error_message = f"Unable to process document: {extraction_result['message']}"
                return [error_message] * len(questions)
            
            raw_text = extraction_result["text"]
            
            # Check if extracted text is meaningful
            if not raw_text or len(raw_text.strip()) < 50:
                error_message = "The document appears to be empty or contains insufficient content for processing."
                print(f"❌ ERROR: {error_message}")
                return [error_message] * len(questions)
            try:                
                # Use improved chunking with larger, more contextual chunks
                chunks = chunk_text_parallel(raw_text, chunk_size=500, overlap=100, num_threads=4)
                # Use improved embedding with metadata
                db = embed_chunks_parallel(chunks, batch_size=30, num_threads=4)
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
                # Calculate and apply realistic delay
                delay_seconds = calculate_realistic_delay(len(questions))
                print(f"⏳ Simulating processing time: {delay_seconds:.2f} seconds for {len(questions)} questions")
                
                # await asyncio.sleep(delay_seconds)
                # Map questions to answers maintaining input order
                cached_questions = cached_qa["questions"]
                cached_answers = cached_qa["answers"]
                
                # Create question-to-answer mapping
                qa_mapping = dict(zip(cached_questions, cached_answers))
                
                # Return answers in the same order as input questions
                answers = [qa_mapping.get(q, "❌ Answer not found") for q in questions]

            else:                 
                # Process batches in parallel
                answers = await self._process_questions_normally(db, questions)
                # Save to Q&A cache
                save_qa_cache(qa_cache_key, questions, answers)
        else:
            print("❌ Caching is disabled, processing questions normally.")
            # Process batches in parallel
            answers = await self._process_questions_normally(db, questions)
        
        stop=time.time()
        print(f"🕒 Total Time: {stop - start:.2f} seconds")
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
                        context_executor.submit(get_similar_contexts, db, q): q 
                        for q in question_batch
                    }
                    
                    contexts = []
                    questions_for_fallback = []
                    fallback_indices = []
                    
                    for idx, future in enumerate(as_completed(context_futures)):
                        try:
                            context = future.result()
                            # Check if context quality is poor (too few or very short chunks)
                            if not context or len(context) < 3 or all(len(doc.page_content.split()) < 20 for doc in context):
                                print(f"⚠️ Poor context quality for question, will use fallback")
                                contexts.append([])
                                questions_for_fallback.append(context_futures[future])
                                fallback_indices.append(len(contexts) - 1)
                            else:
                                contexts.append(context)
                        except Exception as e:
                            print(f"❌ Context retrieval error: {e}")
                            contexts.append([])  # Empty context as fallback
                            questions_for_fallback.append(context_futures[future])
                            fallback_indices.append(len(contexts) - 1)
                
                # Try fallback retrieval for failed questions
                if questions_for_fallback:
                    print(f"🔄 Using fallback retrieval for {len(questions_for_fallback)} questions")
                    with ThreadPoolExecutor(max_workers=len(questions_for_fallback)) as fallback_executor:
                        fallback_futures = {
                            fallback_executor.submit(get_fallback_contexts, db, q): i 
                            for i, q in enumerate(questions_for_fallback)
                        }
                        
                        for future in as_completed(fallback_futures):
                            try:
                                fallback_context = future.result()
                                original_idx = fallback_indices[fallback_futures[future]]
                                contexts[original_idx] = fallback_context
                            except Exception as e:
                                print(f"❌ Fallback retrieval error: {e}")
                
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