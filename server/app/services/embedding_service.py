import faiss
import numpy as np
import os
from typing import List, Dict
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Initialize sentence transformer model
embedding_model = None

# Global variables for embeddings storage
embedding_storage = {}
text_storage = {}

def get_embedding_model():
    """Initialize and return the embedding model"""
    global embedding_model
    if embedding_model is None:
        # Using a high-quality sentence transformer model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded sentence transformer model: all-MiniLM-L6-v2")
    return embedding_model

async def create_embeddings(document_chunks: List[Dict], execution_id: str):
    """
    Create embeddings for document chunks using Sentence Transformers
    """
    try:
        model = get_embedding_model()
        texts = [chunk["text"] for chunk in document_chunks]
        
        # Create embeddings using Sentence Transformers
        embeddings = model.encode(texts, convert_to_numpy=True)
        embeddings_array = embeddings.astype('float32')
        
        # Normalize embeddings for better similarity search
        faiss.normalize_L2(embeddings_array)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for similarity
        index.add(embeddings_array)
        
        # Store in memory (in production, use proper vector DB)
        embedding_storage[execution_id] = index
        text_storage[execution_id] = document_chunks
        
        logger.info(f"Created {len(embeddings)} embeddings for execution {execution_id}")
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

async def search_similar(query: str, execution_id: str, k: int = 5) -> List[Dict]:
    """
    Search for similar chunks using query embedding
    """
    try:
        model = get_embedding_model()
        
        # Get query embedding
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        index = embedding_storage.get(execution_id)
        if not index:
            raise ValueError(f"No embeddings found for execution {execution_id}")
        
        scores, indices = index.search(query_embedding, k)
        
        # Return relevant chunks
        chunks = text_storage.get(execution_id, [])
        results = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                result = chunks[idx].copy()
                result["similarity_score"] = float(scores[0][i])
                results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching similar chunks: {str(e)}")
        raise