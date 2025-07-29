import openai
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Global variables for embeddings storage
embedding_storage = {}
text_storage = {}

async def create_embeddings(document_chunks: List[Dict], execution_id: str):
    """
    Create embeddings for document chunks using OpenAI
    """
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        texts = [chunk["text"] for chunk in document_chunks]
        
        # Create embeddings using OpenAI
        response = await openai.Embedding.acreate(
            input=texts,
            model="text-embedding-3-large"
        )
        
        embeddings = [item["embedding"] for item in response["data"]]
        embeddings_array = np.array(embeddings).astype('float32')
        
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
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Get query embedding
        response = await openai.Embedding.acreate(
            input=[query],
            model="text-embedding-3-large"
        )
        
        query_embedding = np.array([response["data"][0]["embedding"]]).astype('float32')
        
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