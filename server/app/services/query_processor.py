import openai
import os
from typing import List
import logging
from app.services.embedding_service import search_similar

logger = logging.getLogger(__name__)

async def process_queries(questions: List[str], execution_id: str) -> List[str]:
    """
    Process list of questions and return answers
    """
    answers = []
    
    for question in questions:
        try:
            answer = await process_single_query(question, execution_id)
            answers.append(answer)
        except Exception as e:
            logger.error(f"Error processing query '{question}': {str(e)}")
            answers.append(f"Error processing query: {str(e)}")
    
    return answers

async def process_single_query(question: str, execution_id: str) -> str:
    """
    Process a single query using RAG approach
    """
    try:
        # Step 1: Retrieve relevant chunks
        relevant_chunks = await search_similar(question, execution_id, k=5)
        
        # Step 2: Prepare context from relevant chunks
        context = "\n\n".join([
            f"Chunk {i+1}:\n{chunk['text']}" 
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        # Step 3: Generate answer using LLM
        answer = await generate_answer_with_llm(question, context)
        
        return answer
        
    except Exception as e:
        logger.error(f"Error processing single query: {str(e)}")
        raise

async def generate_answer_with_llm(question: str, context: str) -> str:
    """
    Generate answer using OpenAI GPT with context
    """
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        system_prompt = """You are an expert insurance policy analyst. Based on the provided policy document context, answer the user's question accurately and concisely. 

Guidelines:
1. Use only information from the provided context
2. If the answer isn't in the context, say so clearly
3. Provide specific details like waiting periods, coverage limits, conditions
4. Be precise about exclusions and limitations
5. Cite relevant policy sections when possible"""

        user_prompt = f"""Context from policy document:
{context}

Question: {question}

Please provide a clear, accurate answer based on the policy information above."""

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating LLM answer: {str(e)}")
        raise