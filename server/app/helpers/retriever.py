def get_similar_contexts(vector_store, question: str, k: int = 8):
    return vector_store.similarity_search(question, k=5, fetch_k=10)
