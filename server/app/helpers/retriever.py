

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

def get_similar_contexts(vector_store, question: str, k: int = 15):
    enhanced_question = enhance_query_for_fraud(question)
    enhanced_question = enhance_query_for_contact_details(enhanced_question)
    if enhanced_question != question:
        print(f"🔍 DEBUG: Enhanced question for retrieval: {enhanced_question}")
    return vector_store.similarity_search(question, k=15, fetch_k=25)
