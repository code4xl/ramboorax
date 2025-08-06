from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import faiss
import math

# Keep original for compatibility
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 128}
)

# Direct SentenceTransformer model (matches your working code)
direct_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class EnhancedFAISSVectorStore:
    """Enhanced FAISS vector store that combines LangChain compatibility with direct FAISS performance"""
    
    def __init__(self, chunks):
        self.chunks = chunks
        
        # Create both LangChain and direct FAISS implementations
        print("🔄 Creating enhanced vector store...")
        
        # 1. LangChain version for compatibility
        docs = [Document(page_content=chunk) for chunk in chunks]
        self.langchain_db = FAISS.from_documents(docs, embedding_model)
        
        # 2. Direct FAISS version for better performance
        print("🚀 Creating direct FAISS index...")
        embeddings = direct_embedding_model.encode(chunks, show_progress_bar=True)
        
        # Build direct FAISS index
        dim = embeddings.shape[1]
        self.direct_index = faiss.IndexFlatL2(dim)
        self.direct_index.add(np.array(embeddings, dtype=np.float32))
        
        # Store embeddings and model for search
        self.embeddings = embeddings
        self.model = direct_embedding_model
        
        print(f"✅ Enhanced vector store created with {len(chunks)} chunks")
    
    def similarity_search(self, query: str, k: int = 5, use_direct: bool = True):
        """Search using either direct FAISS or LangChain wrapper"""
        
        if use_direct:
            return self._direct_search(query, k)
        else:
            return self.langchain_db.similarity_search(query, k)
    
    def _direct_search(self, query: str, k: int = 5):
        """Direct FAISS search (matches your working standalone code)"""
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.direct_index.search(
            np.array(query_embedding, dtype=np.float32), k
        )
        
        # Convert to LangChain Documents
        docs = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:
                similarity_score = 1 / (1 + dist)
                doc = Document(
                    page_content=self.chunks[idx],
                    metadata={
                        'similarity_score': similarity_score,
                        'distance': float(dist),
                        'rank': i + 1
                    }
                )
                docs.append(doc)
        
        return docs
    
    # LangChain compatibility methods
    @property
    def index(self):
        return self.langchain_db.index
    
    @property
    def docstore(self):
        return self.langchain_db.docstore
    
    @property
    def index_to_docstore_id(self):
        return self.langchain_db.index_to_docstore_id
    
    def merge_from(self, other):
        """Merge another vector store"""
        return self.langchain_db.merge_from(other.langchain_db if hasattr(other, 'langchain_db') else other)

def embed_chunks(chunks):
    """Standard embedding function"""
    docs = [Document(page_content=chunk) for chunk in chunks]
    db = FAISS.from_documents(docs, embedding_model)
    return db

def embed_chunks_enhanced(chunks):
    """Enhanced embedding function that creates better vector store"""
    return EnhancedFAISSVectorStore(chunks)

def embed_chunks_parallel(chunks, batch_size: int = 50, num_threads: int = 4, use_enhanced: bool = True):
    """
    Create embeddings in parallel batches for faster processing
    """
    
    # If chunks are small, use enhanced single-threaded approach
    if len(chunks) < 100:
        if use_enhanced:
            return embed_chunks_enhanced(chunks)
        else:
            return embed_chunks(chunks)
    
    if use_enhanced:
        print("🚀 Using enhanced parallel embedding...")
        return embed_chunks_enhanced(chunks)
    
    # Original parallel approach for compatibility
    from langchain_community.vectorstores import FAISS
    from langchain.docstore.document import Document
    
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