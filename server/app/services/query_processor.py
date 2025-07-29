import google.generativeai as genai
import os
from typing import List
import logging
from app.services.embedding_service import search_similar
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
gemini_model = None

def get_gemini_model():
    """Initialize and return the Gemini model"""
    global gemini_model
    if gemini_model is None:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Initialized Gemini model: gemini-1.5-flash")
    return gemini_model

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
        
        # Step 3: Generate answer using Gemini
        answer = await generate_answer_with_gemini(question, context)
        
        return answer
        
    except Exception as e:
        logger.error(f"Error processing single query: {str(e)}")
        raise

async def generate_answer_with_gemini(question: str, context: str) -> str:
    """
    Generate answer using Google Gemini with context
    """
    try:
        model = get_gemini_model()
        
        prompt = f"""You are an expert insurance policy analyst. Based on the provided policy document context, answer the user's question accurately and concisely.

Guidelines:
1. Use only information from the provided context
2. If the answer isn't in the context, say so clearly
3. Provide specific details like waiting periods, coverage limits, conditions
4. Be precise about exclusions and limitations
5. Cite relevant policy sections when possible

Context from policy document:
{context}

Question: {question}

Please provide a clear, accurate answer based on the policy information above."""

        # Generate response using Gemini
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500,
                top_p=0.8,
                top_k=10
            )
        )
        
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating Gemini answer: {str(e)}")
        raise