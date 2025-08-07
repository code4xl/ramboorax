import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from app.helpers.embedder import embedding_model
import re
from typing import List, Tuple
from langchain_core.documents import Document

def extract_key_terms(question: str) -> List[str]:
    """Extract key terms that should be emphasized in retrieval"""
    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'a', 'an', 'to', 'for', 'of', 'as', 'by'}
    
    # Extract quoted phrases first
    quoted_phrases = re.findall(r'"([^"]*)"', question)
    
    # Extract capitalized terms (likely names, places, organizations)
    capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
    
    # Extract numbers and dates
    numbers_dates = re.findall(r'\b\d+(?:[,.-]\d+)*\b', question)
    
    # Extract remaining important words
    words = re.findall(r'\b\w+\b', question.lower())
    important_words = [w for w in words if len(w) > 3 and w not in stop_words]
    
    key_terms = quoted_phrases + capitalized_terms + numbers_dates + important_words
    return list(set(key_terms))  # Remove duplicates


def enhance_query_comprehensive(question: str) -> str:
    """Enhanced query expansion based on question type and content"""
    question_lower = question.lower()
    enhanced_parts = [question]
    
    # Contact/Personal Info Enhancement
    if any(term in question_lower for term in ['contact', 'phone', 'email', 'address', 'details']):
        enhanced_parts.append("contact information phone number email address personal details policyholder")
    
    # Financial/Amount Enhancement  
    if any(term in question_lower for term in ['amount', 'premium', 'salary', 'income', 'cost', 'price', 'fee']):
        enhanced_parts.append("amount money payment premium cost financial")
    
    # Location Enhancement
    if any(term in question_lower for term in ['address', 'location', 'pin', 'code', 'city', 'state']):
        enhanced_parts.append("address location city state pin code postal")
    
    # Fraud/Legal Enhancement
    if any(term in question_lower for term in ['fraud', 'forged', 'false', 'illegal', 'violation']):
        enhanced_parts.append("fraud fraudulent violation penalty consequence legal")
    
    # Policy/Document Enhancement
    if any(term in question_lower for term in ['policy', 'document', 'certificate', 'agreement']):
        enhanced_parts.append("policy document certificate agreement terms conditions")
    
    # Date/Time Enhancement
    if any(term in question_lower for term in ['date', 'when', 'time', 'period', 'duration']):
        enhanced_parts.append("date time period duration effective validity")
    
    # Name/Identity Enhancement
    if any(term in question_lower for term in ['name', 'who', 'person', 'individual']):
        enhanced_parts.append("name person individual identity holder")
    
    enhanced_query = " ".join(enhanced_parts)
    
    if enhanced_query != question:
        print(f"🔍 Enhanced query: {enhanced_query}")
    
    return enhanced_query

def get_similar_contexts_old(vector_store, question: str, k: int = 15):
    enhanced_question = enhance_query_comprehensive(question)
    return vector_store.similarity_search(enhanced_question, k=15, fetch_k=25)


def get_similar_contexts(vector_store, question: str, k: int = 15):
    """
    Improved retrieval with hybrid search and re-ranking
    """
    # Step 1: Enhanced query
    enhanced_question = enhance_query_comprehensive(question)
    key_terms = extract_key_terms(question)
    
    # Step 2: Get more candidates for re-ranking
    expanded_k = min(k * 3, 50)  # Get 3x more candidates
    
    # Get the underlying FAISS index and docstore
    faiss_index = vector_store.index
    docstore = vector_store.docstore
    index_to_docstore_id = vector_store.index_to_docstore_id
    
    # Step 3: Semantic search with enhanced query
    query_embedding = embedding_model.embed_query(enhanced_question)
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    # Search using FAISS
    distances, indices = faiss_index.search(query_vector, expanded_k)
    
    # Convert indices to documents with scores
    candidates = []
    for i, distance in zip(indices[0], distances[0]):
        if i != -1:  # Valid index
            doc_id = index_to_docstore_id[i]
            doc = docstore.search(doc_id)
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity_score = 1 / (1 + distance)
            candidates.append((doc, similarity_score))
    
    # Step 4: Re-rank based on key term presence
    def calculate_relevance_score(doc_content: str, base_score: float) -> float:
        content_lower = doc_content.lower()
        
        # Key term bonus
        key_term_score = 0
        for term in key_terms:
            term_lower = term.lower()
            if term_lower in content_lower:
                # Exact phrase match gets higher score
                if len(term.split()) > 1:
                    key_term_score += 0.3
                else:
                    key_term_score += 0.1
                    
        # Length penalty for very short chunks (likely incomplete)
        length_penalty = 0
        if len(doc_content.split()) < 50:
            length_penalty = -0.1
        
        # Bonus for complete sentences
        sentence_bonus = 0
        if doc_content.strip().endswith(('.', '!', '?')):
            sentence_bonus = 0.05
            
        return base_score + key_term_score + length_penalty + sentence_bonus
    
    # Re-rank candidates
    ranked_candidates = []
    for doc, base_score in candidates:
        relevance_score = calculate_relevance_score(doc.page_content, base_score)
        ranked_candidates.append((doc, relevance_score))
    
    # Sort by relevance score (descending)
    ranked_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Step 5: Diversify results to avoid too much redundancy
    final_docs = []
    used_content_signatures = set()
    
    for doc, score in ranked_candidates:
        # Create a signature for the content (first 50 words)
        content_signature = ' '.join(doc.page_content.split()[:50]).lower()
        
        # Check if this content is too similar to already selected docs
        is_too_similar = False
        for existing_sig in used_content_signatures:
            # Simple similarity check
            common_words = len(set(content_signature.split()) & set(existing_sig.split()))
            total_words = len(set(content_signature.split()) | set(existing_sig.split()))
            similarity = common_words / total_words if total_words > 0 else 0
            
            if similarity > 0.6:  # 60% similarity threshold
                is_too_similar = True
                break
        
        if not is_too_similar:
            final_docs.append(doc)
            used_content_signatures.add(content_signature)
            
        if len(final_docs) >= k:
            break
    
    # If we don't have enough diverse results, fill with remaining candidates
    if len(final_docs) < k:
        for doc, score in ranked_candidates:
            if doc not in final_docs:
                final_docs.append(doc)
                if len(final_docs) >= k:
                    break
    
    print(f"🎯 Retrieved {len(final_docs)} relevant chunks for query")
    return final_docs

def get_fallback_contexts(vector_store, question: str, k: int = 15):
    """
    Fallback retrieval using simpler keyword matching when semantic search fails
    """
    print("🔄 Using fallback keyword-based retrieval")
    
    # Extract all documents from vector store
    all_docs = []
    for i in range(vector_store.index.ntotal):
        try:
            doc_id = vector_store.index_to_docstore_id[i]
            doc = vector_store.docstore.search(doc_id)
            all_docs.append(doc)
        except:
            continue
    
    # Simple keyword scoring
    question_words = set(question.lower().split())
    scored_docs = []
    
    for doc in all_docs:
        content_words = set(doc.page_content.lower().split())
        overlap = len(question_words.intersection(content_words))
        score = overlap / len(question_words) if question_words else 0
        scored_docs.append((doc, score))
    
    # Sort by score and return top k
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:k] if score > 0]
