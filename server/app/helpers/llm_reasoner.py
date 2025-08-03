import json
import google.generativeai as genai
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import threading

load_dotenv()

API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"), 
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4")
]
API_KEYS = [key for key in API_KEYS if key]
_key_counter = 0
_key_lock = threading.Lock()

def get_next_api_key():
    global _key_counter
    with _key_lock:
        key = API_KEYS[_key_counter % len(API_KEYS)]
        _key_counter += 1
        return key

def generate_batch_answer(
    contexts: list[list[Document]],
    questions: list[str],
) -> list[str]:
    """
    Sends all questions+contexts in one shot, asks Gemini to reply
    with {"answers": [...]} JSON, then parses it robustly.
    """
    # 1) Build the prompt
    prompt = """You are a document analysis expert. Your task is to extract exact information from the provided context and answer questions ACCURATELY and CONCISELY.

CRITICAL RULES:
1. Use ONLY the provided context for answers—no assumptions or external knowledge.
2. Extract EXACT numbers, dates, percentages, amounts, and terminology from the context.
3. If specific information is not found in context, reply: "Information not available in the provided document."
4. Keep answers under 65 words until and unless specifically asked for explanation.
5. Use PRECISE wording and terminology from the source document.
6. For yes/no questions, start with "Yes" or "No" then provide details.
7. Answers should be easily understandable by non-expert humans.
9. If asked about fraudluent activities, illegal issues or any other sensitive topics and there is nothing related in context, reply with "Information not available in the provided document. This is illegal/fraudulent activity and you should not support it."
10. If question is so worst that it cannot be answered and is incomplete reply whith "This is an incomplete question and cannot be answered."

OUTPUT FORMAT:
Return ONLY valid JSON format (no extra text, explanations, or markdown):
{"answers": ["Answer to Question 1", "Answer to Question 2", ...]}

JSON FORMATTING RULES:
- Use single quotes (') for any quotes needed inside answer text
- Never use double quotes (") inside answers
- Keep answers as single line strings
- Example: {"answers": ["The policy states 'coverage is 80%' for this benefit.", "Yes, this is covered under section 3.2."]}

QUESTIONS AND CONTEXT:
"""
    
    for idx, (q, ctx_docs) in enumerate(zip(questions, contexts), 1):
        ctx_text = "\n".join(d.page_content for d in ctx_docs)
        prompt += (
            f"\nQuestion {idx}: {q}\n"
            f"Context {idx}:\n{ctx_text}\n"
            f"{'='*50}\n"
        )
    
    prompt += "\nIMPORTANT: Return ONLY the JSON object. No additional text or explanations."

    # 2) Call Gemini
    api_key = get_next_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # 3) Clean the response - remove markdown formatting if present
    if raw.startswith("```json"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    # 4) Sanitize: if it's not valid JSON, try to pull out the first {...} block
    json_str = raw
    if not raw.startswith("{"):
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = raw[start : end + 1]

    # 5) Parse
    try:
        payload = json.loads(json_str)
        answers = payload.get("answers")
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise ValueError(f"Wrong format: expected {len(questions)} answers, got {len(answers) if answers else 0}")
        return [str(a).strip() for a in answers]

    except Exception as e:
        print(f"❌ Gemini batch error: {e}")
        print(f"🔍 Raw response: {raw}...")
        
        # 6) Fallback: try to extract from malformed JSON
        try:
            # Sometimes Gemini returns valid JSON but with extra text
            if 'answers' in raw:
                import re
                json_match = re.search(r'\{"answers":\s*\[.*?\]\s*\}', raw, re.DOTALL)
                if json_match:
                    payload = json.loads(json_match.group())
                    answers = payload['answers']
                    while len(answers) < len(questions):
                        answers.append("Information not available")
                    return [str(a).strip() for a in answers[:len(questions)]]
        except:
            pass
            
        # 7) Final fallback: create error responses
        print("🔄 Using fallback: creating error responses")
        return ["Error processing response" for _ in questions]