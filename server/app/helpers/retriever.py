import numpy as np
from app.helpers.embedder import embedding_model


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

def get_similar_contexts_old(vector_store, question: str, k: int = 15):
    enhanced_question = enhance_query_for_fraud(question)
    enhanced_question = enhance_query_for_contact_details(enhanced_question)
    if enhanced_question != question:
        print(f"🔍 DEBUG: Enhanced question for retrieval: {enhanced_question}")
    return vector_store.similarity_search(question, k=15, fetch_k=25)


def get_similar_contexts(vector_store, question: str, k: int = 15):
    # enhanced_question = enhance_query_for_fraud(question)
    # enhanced_question = enhance_query_for_contact_details(enhanced_question)
    # if enhanced_question != question:
    #     print(f"🔍 DEBUG: Enhanced question for retrieval: {enhanced_question}")
    
    # Get the underlying FAISS index and docstore
    faiss_index = vector_store.index
    docstore = vector_store.docstore
    index_to_docstore_id = vector_store.index_to_docstore_id
    
    # Embed the question manually
    query_embedding = embedding_model.embed_query(question)
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    # Search using FAISS
    distances, indices = faiss_index.search(query_vector, k)
    
    # Convert indices to documents
    docs = []
    for i in indices[0]:
        if i != -1:  # Valid index
            doc_id = index_to_docstore_id[i]
            doc = docstore.search(doc_id)
            docs.append(doc)
    
    return docs