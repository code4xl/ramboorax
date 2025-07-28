import httpx
import os
import re
import asyncio
import logging

logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean text by removing unwanted characters and normalizing whitespace"""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\ufeff', '')
    text = text.replace('\u200d', '')
    text = text.replace('\u200c', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_gmail_emails(emails_data):
    """Format Gmail emails for LLM processing"""
    if not emails_data or 'emails' not in emails_data:
        return "No emails found in the input data."
    
    emails = emails_data['emails']
    if not emails:
        return "No emails to process."
    
    formatted_emails = []
    for i, email in enumerate(emails, 1):
        email_text = f"Email {i}:\n"
        email_text += f"ID: {email.get('id', 'N/A')}\n"
        
        # Add subject if available
        if 'subject' in email:
            email_text += f"Subject: {clean_text(email['subject'])}\n"
        
        # Add sender if available
        if 'sender' in email:
            email_text += f"From: {clean_text(email['sender'])}\n"
        
        # Add date if available
        if 'date' in email:
            email_text += f"Date: {email['date']}\n"
        
        # Add snippet/content
        if 'snippet' in email:
            email_text += f"Content: {clean_text(email['snippet'])}\n"
        
        formatted_emails.append(email_text)
    
    return "\n" + "="*50 + "\n".join(formatted_emails) + "="*50 + "\n"

def format_input_data(input_data):
    """Format input data based on its type and source"""
    if not input_data:
        return "No input data provided."
    
    formatted_inputs = []
    
    for i, data in enumerate(input_data):
        if isinstance(data, dict):
            # Check if it's Gmail data
            if 'emails' in data and isinstance(data['emails'], list):
                logger.info(f"Processing Gmail data with {len(data['emails'])} emails")
                formatted_inputs.append(f"Gmail Data:\n{format_gmail_emails(data)}")
            
            # Check if it's Google Calendar data
            elif 'events' in data and isinstance(data['events'], list):
                logger.info(f"Processing Calendar data with {len(data['events'])} events")
                events_text = "\n".join([f"Event: {event.get('summary', 'No title')} - {event.get('start', {}).get('dateTime', 'No date')}" 
                                       for event in data['events']])
                formatted_inputs.append(f"Calendar Data:\n{events_text}")
            
            # Check if it's GitHub data
            elif 'repositories' in data or 'issues' in data or 'pull_requests' in data:
                logger.info("Processing GitHub data")
                formatted_inputs.append(f"GitHub Data:\n{str(data)}")
            
            # Check if it's Supabase/Database data
            elif 'rows' in data or 'data' in data:
                logger.info("Processing Database data")
                formatted_inputs.append(f"Database Data:\n{str(data)}")
            
            # Generic dict handling
            else:
                logger.info("Processing generic dictionary data")
                formatted_inputs.append(f"Data {i+1}:\n{str(data)}")
        
        elif isinstance(data, str):
            # Handle string input (like from customInput nodes)
            logger.info("Processing string input")
            formatted_inputs.append(f"Input {i+1}:\n{clean_text(data)}")
        
        else:
            # Handle other data types
            logger.info(f"Processing {type(data).__name__} input")
            formatted_inputs.append(f"Input {i+1}:\n{str(data)}")
    
    return "\n\n".join(formatted_inputs)

async def call_llm(node_data, input_data):
    """Call LLM with properly formatted input data"""
    model_provider = node_data.get('modelProvider')
    api_key = node_data.get('apiKey', '').strip()
    system_prompt = node_data.get('systemPrompt', '')
    model_name = node_data.get('modelName', 'gpt-4o')
    temperature = node_data.get('temperature', 0.3)
    max_tokens = node_data.get('maxTokens', 2000)
    
    # Validate inputs
    if not api_key:
        raise ValueError("API key is missing")
    
    if not system_prompt:
        raise ValueError("System prompt is missing")
    
    if model_provider == 'OpenAI':
        # Validate OpenAI API key format
        if not api_key.startswith('sk-'):
            raise ValueError(f"Invalid OpenAI API key format: {api_key[:10]}...")
        
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # Format input data based on type
        user_input = format_input_data(input_data)
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        logger.info(f"Making OpenAI API call with model: {model_name}")
        logger.info(f"Input data types: {[type(data).__name__ for data in input_data]}")
        logger.debug(f"Formatted input length: {len(user_input)} characters")
        
        print("=== LLM Request Debug ===")
        print(f"Model: {model_name}")
        print(f"System Prompt: {system_prompt[:100]}...")
        print(f"User Input Preview: {user_input[:200]}...")
        print("========================")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:  # Increased timeout
                response = await client.post(url, headers=headers, json=payload)
                
                # Handle rate limiting
                if response.status_code == 429:
                    logger.warning("Rate limited. Retrying in 10 seconds...")
                    await asyncio.sleep(10)
                    response = await client.post(url, headers=headers, json=payload)
                
                # Handle specific error codes
                if response.status_code == 401:
                    raise ValueError("Invalid OpenAI API key - authentication failed")
                elif response.status_code == 400:
                    error_detail = response.json().get('error', {}).get('message', 'Bad request')
                    raise ValueError(f"OpenAI API error: {error_detail}")
                
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Log usage stats
                usage = result.get('usage', {})
                if usage:
                    logger.info(f"Token usage - Prompt: {usage.get('prompt_tokens')}, "
                              f"Completion: {usage.get('completion_tokens')}, "
                              f"Total: {usage.get('total_tokens')}")
                
                logger.info("LLM request completed successfully")
                return content

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"LLM HTTP Error: {error_msg}")
            return f"LLM Request Failed: {error_msg}"
        
        except httpx.RequestError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(f"LLM Connection Error: {error_msg}")
            return f"LLM Request Failed: {error_msg}"
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"LLM Unexpected Error: {error_msg}")
            return f"LLM Request Failed: {error_msg}"
    
    elif model_provider == 'Anthropic':
        # Add Claude API support if needed
        return "Anthropic/Claude support not implemented yet"
    
    elif model_provider == 'Google':
        # Google Gemini API implementation
        if not api_key:
            raise ValueError("Google API key is missing")
        
        # Default to gemini-1.5-flash if no model specified
        if not model_name:
            model_name = 'gemini-1.5-flash'
        
        # Gemini REST API endpoint
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}'
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Format input data
        user_input = format_input_data(input_data)
        
        # Combine system prompt and user input for Gemini
        combined_prompt = f"{system_prompt}\n\nUser Input:\n{user_input}"
        
        # Gemini API payload structure
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": combined_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.8,
                "topK": 10
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        logger.info(f"Making Google Gemini API call with model: {model_name}")
        logger.info(f"Input data types: {[type(data).__name__ for data in input_data]}")
        logger.debug(f"Combined prompt length: {len(combined_prompt)} characters")
        
        print("=== Gemini Request Debug ===")
        print(f"Model: {model_name}")
        print(f"System Prompt: {system_prompt[:100]}...")
        print(f"User Input Preview: {user_input[:200]}...")
        print("============================")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                # Handle rate limiting
                if response.status_code == 429:
                    logger.warning("Gemini rate limited. Retrying in 10 seconds...")
                    await asyncio.sleep(10)
                    response = await client.post(url, headers=headers, json=payload)
                
                # Handle specific error codes
                if response.status_code == 400:
                    error_response = response.json()
                    error_msg = error_response.get('error', {}).get('message', 'Bad request')
                    raise ValueError(f"Gemini API error: {error_msg}")
                elif response.status_code == 403:
                    raise ValueError("Invalid Google API key or insufficient permissions")
                
                response.raise_for_status()
                result = response.json()
                
                # Extract content from Gemini response
                candidates = result.get('candidates', [])
                if not candidates:
                    return "No response generated by Gemini"
                
                content_parts = candidates[0].get('content', {}).get('parts', [])
                if not content_parts:
                    return "Empty response from Gemini"
                
                content = content_parts[0].get('text', '')
                
                # Log usage stats if available
                usage_metadata = result.get('usageMetadata', {})
                if usage_metadata:
                    logger.info(f"Token usage - Prompt: {usage_metadata.get('promptTokenCount')}, "
                              f"Candidates: {usage_metadata.get('candidatesTokenCount')}, "
                              f"Total: {usage_metadata.get('totalTokenCount')}")
                
                # Check if response was blocked by safety filters
                finish_reason = candidates[0].get('finishReason')
                if finish_reason == 'SAFETY':
                    safety_ratings = candidates[0].get('safetyRatings', [])
                    blocked_categories = [rating['category'] for rating in safety_ratings 
                                        if rating.get('blocked', False)]
                    logger.warning(f"Gemini response blocked by safety filters: {blocked_categories}")
                    return f"Response blocked by safety filters: {', '.join(blocked_categories)}"
                
                logger.info("Gemini request completed successfully")
                return content
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"Gemini HTTP Error: {error_msg}")
            return f"Gemini Request Failed: {error_msg}"
        
        except httpx.RequestError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(f"Gemini Connection Error: {error_msg}")
            return f"Gemini Request Failed: {error_msg}"
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Gemini Unexpected Error: {error_msg}")
            return f"Gemini Request Failed: {error_msg}"
        
    else:
        return f"Unsupported LLM provider: {model_provider}"
