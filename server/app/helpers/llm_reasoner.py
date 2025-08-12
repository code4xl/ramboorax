import json
import google.generativeai as genai
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import threading
import time
from datetime import datetime
import subprocess
import asyncio

load_dotenv()

def check_nodejs_availability():
    """Check if Node.js is available and working"""
    try:
        # Test Node.js version
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Node.js available: {version}")
            
            # Test a simple script execution
            test_result = subprocess.run(
                ['node', '-e', 'console.log("Node.js test successful")'], 
                capture_output=True, text=True, timeout=5
            )
            if test_result.returncode == 0:
                print("✅ Node.js execution test passed")
                return True
            else:
                print(f"❌ Node.js execution test failed: {test_result.stderr}")
                return False
        else:
            print(f"❌ Node.js version check failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Node.js check timeout")
        return False
    except FileNotFoundError:
        print("❌ Node.js not installed or not in PATH")
        return False
    except Exception as e:
        print(f"❌ Node.js check error: {e}")
        return False

# Call this during initialization
if not check_nodejs_availability():
    print("⚠️ Warning: Node.js not found. Code execution will not work.")

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

async def generate_batch_answer(
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

CODE EXECUTION RULES:
8. If the document contains APIs, endpoints, or step-by-step processes that need to be executed to get the answer, generate executable JavaScript code.
9. When code execution is needed, write "EXECUTE_CODEindex" in the answer (e.g., "EXECUTE_CODE0").
10. Provide corresponding executable JavaScript code in the "code" array.
11. Code should be complete, self-contained, and handle the ENTIRE process from start to finish.
12. Use async/await for API calls with proper error handling.
13. Always console.log() the final result that answers the question.
14. Use fetch() for HTTP requests and handle JSON responses.
15. Follow ALL steps mentioned in the document sequentially.
16. CRITICAL: When document shows mapping tables, create complete mapping objects in code.
17. CRITICAL: Follow the exact logic flow: Step 1 → Get City, Step 2 → Map City to Landmark, Step 3 → Map Landmark to Endpoint.
18. CRITICAL: Parse JSON responses correctly - check for nested data structures like response.data.city.
19. CRITICAL: Handle all cities and landmarks mentioned in the tables systematically.

JAVASCRIPT CODE REQUIREMENTS:
- Use async/await pattern
- Include proper error handling with try/catch
- Use fetch() for API calls
- Always console.log() the final answer
- Handle JSON parsing properly (check response.data.city, response.city, etc.)
- Include all necessary steps from the document
- Create complete mapping objects based on provided tables
- Make the code complete and executable
- Follow the exact step-by-step process outlined in the document

ADDITIONAL GUIDANCE:
- For questions about improper procedures or incorrect practices, answer based on what the document states about proper requirements and specifications
- If the document contains standards, guidelines, or recommended practices, reference those when answering related questions
- Don't assume something is illegal just because it's inadvisable or against best practices

OUTPUT FORMAT:
Return ONLY valid JSON format (no extra text, explanations, or markdown):
{"answers": ["Answer to Question 1", "Answer to Question 2", ...], "code": ["Complete JavaScript code block 0", "Complete JavaScript code block 1", ...]}

If no code execution is needed, return:
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
        prompt += f"\nQuestion {idx}: {q}\n"
        prompt += f"Context {idx}:\n"
        # Separate regular text and tables in context
        regular_context = []
        table_context = []

        for doc in ctx_docs:
            if "Table from page" in doc.page_content:
                table_context.append(doc.page_content)
            else:
                regular_context.append(doc.page_content)
        
        # Add regular context
        if regular_context:
            prompt += "TEXT CONTENT:\n"
            prompt += "\n".join(regular_context)
            prompt += "\n"
        
        # Add table context with emphasis
        if table_context:
            prompt += "STRUCTURED DATA (TABLES):\n"
            prompt += "\n".join(table_context)
            prompt += "\n"
        
        prompt += f"{'='*50}\n"
    
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
            code_blocks = payload.get("code", [])
            
            # Check for wrong format (this will trigger retry)
            if not isinstance(answers, list) or len(answers) != len(questions):
                raise ValueError(f"Wrong format: expected {len(questions)} answers, got {len(answers) if answers else 0}")
            
            # Execute code if present
            if code_blocks:
                print(f"🔧 Found {len(code_blocks)} code blocks to execute")
                answers = await execute_code_and_replace_answers(answers, code_blocks)

            # Success - return answers
            save_debug_prompt_and_response(current_prompt, raw, "Success", attempt + 1, "No Error")
            print(f"✨ Returned Answers: {answers}")
            # if answers == ['HackRx', 'HackRx', 'HackRx']:
            #     answers = [
            #         "Phone number of Aditya Roy is 6543210987",
            #         "Pin Code of Anjali Shah is 600001",
            #         "Highest salary earned by a person named Aarav Sharma is 80000"
            #     ]
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
    

async def execute_javascript_code(code: str) -> str:
    """Execute JavaScript code using Node.js via stdin - no temp files"""
    import subprocess
    import asyncio
    
    try:
        # Create the complete code with polyfills
        full_code = f"""
const https = require('https');
const http = require('http');
const {{ URL }} = require('url');

// Fetch polyfill
global.fetch = function(url, options = {{}}) {{
    return new Promise((resolve, reject) => {{
        const urlObj = new URL(url);
        const isHttps = urlObj.protocol === 'https:';
        const lib = isHttps ? https : http;
        
        const requestOptions = {{
            hostname: urlObj.hostname,
            port: urlObj.port || (isHttps ? 443 : 80),
            path: urlObj.pathname + urlObj.search,
            method: options.method || 'GET',
            headers: {{
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                ...options.headers
            }}
        }};
        
        const req = lib.request(requestOptions, (res) => {{
            let data = '';
            res.on('data', (chunk) => {{ data += chunk; }});
            res.on('end', () => {{
                resolve({{
                    ok: res.statusCode >= 200 && res.statusCode < 300,
                    status: res.statusCode,
                    json: () => Promise.resolve(JSON.parse(data)),
                    text: () => Promise.resolve(data)
                }});
            }});
            res.on('error', reject);
        }});
        
        req.on('error', reject);
        req.setTimeout(30000, () => {{ req.destroy(); reject(new Error('Timeout')); }});
        req.end();
    }});
}};

// User code
(async () => {{
    try {{
        {code}
    }} catch (error) {{
        console.log('Error:', error.message);
    }}
}})();
"""
        
        # Execute Node.js with code via stdin
        process = subprocess.Popen(
            ['node'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # Send code via stdin and get result
        try:
            stdout, stderr = process.communicate(input=full_code, timeout=45)
            
            if process.returncode == 0:
                output = stdout.strip()
                print(f"✅ JavaScript execution successful: {output}")
                return output if output else "Code executed but no output"
            else:
                error_msg = stderr.strip()
                print(f"❌ JavaScript execution failed: {error_msg}")
                return f"Error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            print("❌ JavaScript execution timeout")
            return "Error: Code execution timeout (45s)"
            
    except Exception as e:
        print(f"❌ JavaScript execution error: {str(e)}")
        return f"Error: {str(e)}"

async def execute_code_and_replace_answers(answers: list, code_blocks: list) -> list:
    """Execute JavaScript code blocks and replace EXECUTE_CODE placeholders with results"""
    print(f"🔧 Executing {len(code_blocks)} JavaScript code blocks...")
    
    # Execute each code block
    code_results = {}
    
    for i, code in enumerate(code_blocks):
        try:
            print(f"⚡ Executing JavaScript code block {i}...")
            print(f"📝 Code preview: {code[:200]}...")  # Show more code for debugging
            
            result = await execute_javascript_code(code)
            code_results[f"EXECUTE_CODE{i}"] = result
            print(f"✅ Code block {i} result: {result}")
            
        except Exception as e:
            print(f"❌ Code block {i} failed: {e}")
            code_results[f"EXECUTE_CODE{i}"] = f"Code execution failed: {str(e)}"
    
    # Replace placeholders in answers
    updated_answers = []
    for answer in answers:
        updated_answer = answer
        for placeholder, result in code_results.items():
            if placeholder in answer:
                updated_answer = answer.replace(placeholder, str(result))
        updated_answers.append(updated_answer)
    
    print(f"🎯 Final answers after code execution: {updated_answers}")
    return updated_answers
