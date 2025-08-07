from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor
import math

# Use a more powerful embedding model for better document retrieval
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Better for document retrieval
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 64, "normalize_embeddings": True}  # Normalize for better similarity
)

def embed_chunks(chunks):
    """Create embeddings with improved metadata for better retrieval"""
    docs = []
    for i, chunk in enumerate(chunks):
        # Add chunk metadata for better context
        metadata = {
            "chunk_id": i,
            "chunk_length": len(chunk.split()),
            "chunk_start": chunk[:100] + "..." if len(chunk) > 100 else chunk
        }
        docs.append(Document(page_content=chunk, metadata=metadata))
    
    db = FAISS.from_documents(docs, embedding_model)
    return db

def embed_chunks_parallel(chunks, batch_size: int = 50, num_threads: int = 4):
    """
    Create embeddings in parallel batches for faster processing
    """
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    
    # If chunks are small, use regular embedding
    if len(chunks) < 100:
        return embed_chunks(chunks)
    
    # Split chunks into batches
    chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    def create_batch_embeddings(batch_chunks):
        docs = []
        for i, chunk in enumerate(batch_chunks):
            # Add metadata for each chunk
            metadata = {
                "chunk_id": len(docs),  # Global chunk ID across batches
                "chunk_length": len(chunk.split()),
                "chunk_start": chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
            docs.append(Document(page_content=chunk, metadata=metadata))
        return FAISS.from_documents(docs, embedding_model)
    
    # Process batches in parallel
    vector_stores = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(create_batch_embeddings, batch) for batch in chunk_batches]
        
        for future in futures:
            vs = future.result()
            vector_stores.append(vs)
    
    # Merge all vector stores
    if len(vector_stores) == 1:
        return vector_stores[0]
    
    main_vs = vector_stores[0]
    for vs in vector_stores[1:]:
        main_vs.merge_from(vs)
    
    return main_vs