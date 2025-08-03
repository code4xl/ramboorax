from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor
import math

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 128}

)

def embed_chunks(chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
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
        docs = [Document(page_content=chunk) for chunk in batch_chunks]
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