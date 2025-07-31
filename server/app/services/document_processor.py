from app.helpers.processor import extract_text_from_url, chunk_text_parallel, chunk_text
from app.helpers.embedder import embed_chunks_parallel, embed_chunks
from app.helpers.retriever import get_similar_contexts
from app.helpers.llm_reasoner import generate_batch_answer
from app.helpers.cache_manager import load_vector_store_if_exists, save_vector_store
import time

class DocumentProcessorService:
    async def process_document_and_questions(self, document_url: str, questions: list) -> list:
        start=time.time()
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
            raw_text = extract_text_from_url(document_url)
            # chunks = chunk_text(raw_text)
            chunks = chunk_text_parallel(raw_text, num_threads=4)
            # db = embed_chunks(chunks)
            db = embed_chunks_parallel(chunks, batch_size=50, num_threads=4)
            save_vector_store(db, document_url)

        # Batch Question Processing
        batch_size = 5
        answers = []

        for i in range(0, len(questions), batch_size):
            question_batch = questions[i:i + batch_size]
            contexts = [get_similar_contexts(db, q) for q in question_batch]
            batch_answers = generate_batch_answer(contexts, question_batch)
            answers.extend(batch_answers)
        
        stop=time.time()
        print(f"🕒 Total Time: {stop - start:.2f} seconds")
        return answers