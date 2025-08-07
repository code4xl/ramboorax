import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from app.helpers.embedder import embedding_model

class DirectFAISSRetriever:
    """Direct FAISS retriever that matches your working standalone code"""
    
    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        
        # Create embeddings
        print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
        self.embeddings = self.model.encode(chunks, show_progress_bar=True)
        
        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings, dtype=np.float32))
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
    
    def search_similar_chunks(self, query: str, top_k: int = 5):
        """Search for similar chunks using direct FAISS"""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        # Return chunks with similarity scores
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid index
                similarity_score = 1 / (1 + dist)  # Convert L2 distance to similarity
                results.append({
                    'chunk': self.chunks[idx],
                    'distance': dist,
                    'similarity': similarity_score,
                    'rank': i + 1
                })
        
        return results

def enhance_query_for_fraud(question: str) -> str:
    fraud_keywords = ['fraud', 'forged', 'false statement', 'fraudulent means', 'fabricated']
    if any(keyword in question.lower() for keyword in fraud_keywords):
        return question + " fraud policy violation penalties consequences"
    return question

def enhance_query_for_contact_details(question: str) -> str:
    contact_terms = ['contact', 'policyholder']
    
    if any(term in question.lower() for term in contact_terms):
        return question + " insured detail name address phone email agent contact policy"
    return question

def get_similar_contexts_enhanced(vector_store, question: str, k: int = 5):
    """Enhanced retrieval using direct FAISS approach"""
    
    # First try with the enhanced query approach from your working code
    enhanced_question = enhance_query_for_fraud(question)
    enhanced_question = enhance_query_for_contact_details(enhanced_question)
    
    if enhanced_question != question:
        print(f"🔍 DEBUG: Enhanced question for retrieval: {enhanced_question}")
    
    # Check if we have a direct FAISS retriever stored
    if hasattr(vector_store, '_direct_retriever'):
        print("🚀 Using direct FAISS retriever")
        results = vector_store._direct_retriever.search_similar_chunks(enhanced_question, k)
        
        # Convert back to LangChain Document format
        docs = []
        for result in results:
            doc = Document(
                page_content=result['chunk'],
                metadata={
                    'similarity_score': result['similarity'],
                    'distance': result['distance'],
                    'rank': result['rank']
                }
            )
            docs.append(doc)
        
        # Debug output
        print(f"📊 DEBUG: Retrieved {len(docs)} chunks with direct FAISS")
        # for i, doc in enumerate(docs[:3]):
        #     print(f"🎯 Chunk {i+1} (sim: {doc.metadata.get('similarity_score', 0):.3f}): {doc.page_content[:150]}...")
        
        return docs
    
    else:
        # print("⚠️ Falling back to LangChain FAISS wrapper")
        # Fallback to original method but with better parameters
        return get_similar_contexts_original(vector_store, enhanced_question, k)

def get_similar_contexts_original(vector_store, question: str, k: int = 5):
    """Original method with optimized parameters"""
    
    # Get the underlying FAISS index and docstore
    faiss_index = vector_store.index
    docstore = vector_store.docstore
    index_to_docstore_id = vector_store.index_to_docstore_id
    
    # Embed the question manually using the same model
    query_embedding = embedding_model.embed_query(question)
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    # Search using FAISS with smaller k for better precision
    distances, indices = faiss_index.search(query_vector, k)
    
    # Convert indices to documents with similarity scores
    docs = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx != -1:  # Valid index
            doc_id = index_to_docstore_id[idx]
            doc = docstore.search(doc_id)
            
            # Add similarity metadata
            similarity_score = 1 / (1 + dist)
            doc.metadata.update({
                'similarity_score': similarity_score,
                'distance': float(dist),
                'rank': i + 1
            })
            docs.append(doc)
    
    # Debug output
    # print(f"📊 DEBUG: Retrieved {len(docs)} chunks with LangChain wrapper")
    # for i, doc in enumerate(docs[:3]):
    #     print(f"🎯 Chunk {i+1} (sim: {doc.metadata.get('similarity_score', 0):.3f}): {doc.page_content[:150]}...")
    
    return docs

def get_similar_contexts(vector_store, question: str, k: int = 5):
    """Main retrieval function - uses enhanced method"""
    return get_similar_contexts_enhanced(vector_store, question, k)