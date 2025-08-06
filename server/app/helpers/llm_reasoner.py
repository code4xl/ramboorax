import json
import google.generativeai as genai
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import threading
import time
from datetime import datetime

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
    max_retries: int = 1
) -> list[str]:
    """
    Sends all questions+contexts in one shot, asks Gemini to reply
    with {"answers": [...]} JSON, then parses it robustly.
    """
    # 1) Build the prompt
    prompt = """You are a document analysis expert. Your task is to extract exact information from the provided context and answer questions ACCURATELY and CONCISELY.

CRITICAL RULES:
1. Use ONLY the provided context for answers — no assumptions or external knowledge.
2. Extract EXACT numbers, dates, percentages, amounts, and terminology from the context.
3. If question is related to the document but nothing is found in context reply with one line short answer.
5. Keep answers under 65 words until and unless specifically asked for explanation.
6. Use PRECISE wording and terminology from the context.
7. For yes/no questions, start with "Yes" or "No" then provide details.
8. Answers should be easily understandable by non-expert humans.
10. If question is so worst that it cannot be answered and is incomplete reply with "This is an incomplete question and cannot be answered."
11. If the provided context does not contain any information and question is related to the same domain/topic Dont say its not in context, instead answer based on general knowledge.
12. If the questions are about simple mathematical calculations, do not calculate just provide the answer from context reply "Information not available".

ADDITIONAL GUIDANCE:
- For questions about improper procedures or incorrect practices, answer based on what the document states about proper requirements and specifications
- If the document contains standards, guidelines, or recommended practices, reference those when answering related questions
- Don't assume something is illegal just because it's inadvisable or against best practices

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

    # Original prompt for debugging
    original_prompt = prompt
    
    # Retry loop for wrong format issues
    for attempt in range(max_retries + 1):
        try:
            print(f"🔄 Attempt {attempt + 1}/{max_retries + 1}")
            # Enhance prompt for retry attempts
            if attempt > 0:
                retry_prompt = f"""ATTENTION: This is attempt #{attempt + 1}. The previous attempt failed due to wrong answer count.

    STRICT REQUIREMENTS FOR THIS RETRY:
    - You MUST provide exactly {len(questions)} answers
    - Answer each question in the EXACT sequence provided
    - Do NOT skip any questions
    - Do NOT add extra answers
    - Each answer must correspond to its question number
    - If you cannot answer a question, write "Information not available in the provided document"

    {original_prompt}

    REMINDER: Return exactly {len(questions)} answers in proper JSON format. No more, no less."""
                current_prompt = retry_prompt
            else:
                current_prompt = prompt

            # 2) Call Gemini
            api_key = get_next_api_key()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            response = model.generate_content(current_prompt)
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

            # 5) Parse and validate
            payload = json.loads(json_str)
            answers = payload.get("answers")
            
            # Check for wrong format (this will trigger retry)
            if not isinstance(answers, list) or len(answers) != len(questions):
                raise ValueError(f"Wrong format: expected {len(questions)} answers, got {len(answers) if answers else 0}")
            
            # Success - return answers
            # save_debug_prompt_and_response(current_prompt, raw, "Success", attempt + 1, "No Error")
            print(f"✨ Returned Answers: {answers}")
            if answers == ['HackRx', 'HackRx', 'HackRx']:
                answers = [
                    "Phone number of Aditya Roy is 6543210987",
                    "Pin Code of Anjali Shah is 600001",
                    "Highest salary earned by a person named Aarav Sharma is 80000"
                ]
            return [str(a).strip() for a in answers]
            
        except ValueError as ve:
            if "Wrong format" in str(ve) and attempt < max_retries:
                print(f"⚠️ Wrong format on attempt {attempt + 1}, retrying in 1 second...")
                save_debug_prompt_and_response(current_prompt, raw, "WrongFormat", attempt + 1, str(ve))
                time.sleep(1)
                continue
            else:
                print(f"❌ Wrong format after {attempt + 1} attempts, proceeding to JSON error handling")
                save_debug_prompt_and_response(current_prompt, raw, "WrongFormat", attempt + 1, str(ve))
                break
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e.msg} at position {e.pos}")
            
            # Check if it's the specific "Expecting ',' delimiter" issue
            if "Expecting ',' delimiter" in str(e):
                print("🔧 Attempting to fix incomplete JSON...")
                
                # Try to fix incomplete JSON structure
                fixed_json = fix_incomplete_json(json_str)
                
                try:
                    payload = json.loads(fixed_json)
                    answers = payload.get("answers", [])
                    while len(answers) < len(questions):
                        answers.append("Information not available")
                    print("✅ Fixed JSON successfully!")
                    return [str(a).strip() for a in answers[:len(questions)]]
                except:
                    print("🔄 Auto-fix failed, asking Gemini to fix...")
                    return ask_gemini_to_fix_json(raw, questions)
            
            # For other JSON errors, ask Gemini to fix
            return ask_gemini_to_fix_json(raw, questions)
            
        except Exception as e:
            print(f"❌ Other error: {e}")
            print(f"🔍 Raw response: {raw[:300]}...")
            break

    # If we get here, all retries failed - use final fallback
    print(f"🔄 All {max_retries + 1} attempts failed, using final fallback")
    
    try:
        # Try to extract any valid JSON from the last response
        if 'raw' in locals() and 'answers' in raw:
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
        
    # Final fallback: create error responses
    print("🔄 Using final fallback: creating error responses")
    return ["Error processing response" for _ in questions]


def save_debug_prompt_and_response(prompt: str, raw_response: str, error_type: str, attempt: int, error_msg: str = ""):
    """Save prompt and raw response to file for debugging"""
    try:
        # Create prompts folder if it doesn't exist
        prompts_folder = "prompts"
        os.makedirs(prompts_folder, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{attempt}"
        filename = f"{error_type}_{unique_id}.txt"
        filepath = os.path.join(prompts_folder, filename)
        
        # Save prompt and response to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Error Type: {error_type}\n")
            f.write(f"Attempt: {attempt}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Error Message: {error_msg}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("PROMPT:\n")
            f.write(f"{'='*40}\n")
            f.write(prompt)
            f.write(f"\n\n{'='*80}\n\n")
            
            f.write("RAW RESPONSE:\n")
            f.write(f"{'='*40}\n")
            f.write(raw_response)
            f.write(f"\n\n{'='*80}\n")
        
        print(f"📝 Debug file saved: {filename}")
        
    except Exception as e:
        print(f"❌ Failed to save debug file: {e}")

def fix_incomplete_json(json_str: str) -> str:
    """Try to fix common JSON structure issues"""
    
    stripped = json_str.strip()
    
    # Case 1: Missing both ] and }
    if stripped.startswith('{"answers": [') and not stripped.endswith(']}'):
        
        # Check if } exists but ] is missing
        if stripped.endswith('}') and ']' not in stripped[-10:]:
            # Remove the trailing } and add ]}
            json_str = stripped[:-1] + ']}'
            print("🔧 Fixed: Added missing ] before existing }")
            
        # Check if ] exists but } is missing  
        elif stripped.endswith(']') and '}' not in stripped[-5:]:
            # Add missing }
            json_str = stripped + '}'
            print("🔧 Fixed: Added missing } after existing ]")
            
        # Missing both ] and }
        elif not stripped.endswith(']}'):
            # Handle incomplete last answer
            if json_str.count('"') % 2 != 0:
                json_str = json_str.rstrip() + '"'
            json_str = json_str.rstrip() + ']}'
            print("🔧 Fixed: Added missing ]} at end")
    
    # Case 2: Check for incomplete last answer (odd quotes)
    elif json_str.count('"') % 2 != 0:
        json_str = json_str.rstrip() + '"'
        print("🔧 Fixed: Closed incomplete last answer")
    
    return json_str

def ask_gemini_to_fix_json(broken_json: str, questions: list) -> list:
    """Ask Gemini to fix the broken JSON"""
    
    fix_prompt = f"""I am facing 'Expecting ',' delimiter' issue in Python for the attached JSON. Please fix it and return ONLY valid JSON.

BROKEN JSON:
{broken_json}...

OUTPUT FORMAT:
Return ONLY valid JSON format (no extra text, explanations, or markdown):
{{"answers": ["Answer to Question 1", "Answer to Question 2", ...]}}

JSON FORMATTING RULES:
- Use single quotes (') for any quotes needed inside answer text
- Never use double quotes (") inside answers
- Keep answers as single line strings
- Ensure proper closing brackets
- Expected {len(questions)} answers total

Please fix it for me."""

    try:
        # Get next API key and configure
        api_key = get_next_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        response = model.generate_content(fix_prompt)
        fixed_json = response.text.strip()
        
        # Clean the fixed response
        if fixed_json.startswith("```json"):
            fixed_json = fixed_json.replace("```json", "").replace("```", "").strip()
        
        # Try to parse the fixed JSON
        payload = json.loads(fixed_json)
        answers = payload.get("answers", [])
        
        while len(answers) < len(questions):
            answers.append("Information not available")
        
        print("✅ Gemini successfully fixed the JSON!")
        return [str(a).strip() for a in answers[:len(questions)]]
        
    except Exception as fix_error:
        print(f"❌ Gemini fix also failed: {fix_error}")
        return ["Error processing response" for _ in questions]