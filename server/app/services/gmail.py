import httpx
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

async def refresh_token(credentials):
    """Refresh the access token using refresh token"""
    refresh_url = "https://oauth2.googleapis.com/token"
    
    # You need to add your client_secret here (get it from Google Cloud Console)
    CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")  # Replace with actual secret
    # print(credentials["refresh_token"])
    # print("Client Secret: ", CLIENT_SECRET)
    
    data = {
        "client_id": credentials["client_id"],
        "client_secret": CLIENT_SECRET,
        "refresh_token": credentials["refresh_token"],
        "grant_type": "refresh_token"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(refresh_url, data=data)
        if response.status_code == 200:
            token_data = response.json()
            # print(token_data)
            return token_data["access_token"]
        else:
            raise Exception(f"Token refresh failed: {response.text}")


async def fetch_emails(gmail_credentials, input_data):

    credentials = gmail_credentials['credentials']
    print("inside fetch_mails...")
    try:
        new_access_token = await refresh_token(credentials)
        print(f"New access token: {new_access_token}")
        # Update the auth header with new token
        auth_header = f"Bearer {new_access_token}"
    except Exception as e:
        print(f"Token refresh failed: {e}")
        # Fallback to existing token
        auth_header = gmail_credentials['execution_context']['auth_header']
    
    access_token = gmail_credentials['credentials']['access_token']
    user_id = gmail_credentials['execution_context']['user_id']
    base_url = gmail_credentials['execution_context']['base_url']
    # auth_header = gmail_credentials['execution_context']['auth_header']
    # print(f"Updated access token: {auth_header}")
    # Example: Fetch unread emails from yesterday
    query = 'is:unread newer_than:1d'
    url = f'{base_url}/users/{user_id}/messages?q={query}'

    headers = {
        'Authorization': auth_header,
        'Accept': 'application/json'
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        messages = data.get('messages', [])
        print(f"Fetched {len(messages)} messages from Gmail for user {user_id}")

        # Fetch message details
        emails = []
        for message in messages[:10]:  # Limit to 10 emails for now
            message_id = message['id']
            msg_url = f'{base_url}/users/{user_id}/messages/{message_id}'
            msg_response = await client.get(msg_url, headers=headers)
            msg_response.raise_for_status()
            msg_data = msg_response.json()

            snippet = msg_data.get('snippet', '')
            emails.append({'id': message_id, 'snippet': snippet})
        print(f"Fetched {len(emails)} email snippets")
        print(emails)

        return {'emails': emails, 'fetched_at': datetime.now(timezone.utc).isoformat()}


async def send_emails(gmail_credentials, input_data):
    """Send emails dynamically based on LLM output"""
    credentials = gmail_credentials['credentials']
    
    try:
        new_access_token = await refresh_token(credentials)
        auth_header = f"Bearer {new_access_token}"
    except Exception as e:
        print(f"Token refresh failed: {e}")
        auth_header = gmail_credentials['execution_context']['auth_header']
    
    user_id = gmail_credentials['execution_context']['user_id']
    base_url = gmail_credentials['execution_context']['base_url']
    
    # Parse the input data from LLM
    if not input_data or len(input_data) == 0:
        return {"error": "No input data provided"}
    
    llm_output = input_data[0]  # Get LLM output
    
    # Handle string JSON input
    if isinstance(llm_output, str):
        try:
            import json
            llm_output = json.loads(llm_output)
        except:
            return {"error": "Invalid JSON format in LLM output"}
    
    # Validate LLM output structure
    if not isinstance(llm_output, dict):
        return {"error": "LLM output must be a dictionary/object"}
    
    sent_emails = []
    
    # Process each email task from LLM output
    for task_key, task_data in llm_output.items():
        if task_key in ['metadata', 'timestamp', 'fetched_at']:  # Skip metadata
            continue
            
        try:
            # Extract email details from LLM output
            email_details = extract_email_details(task_data)
            
            if not email_details:
                print(f"Skipping {task_key}: No valid email details found")
                continue
            
            # Send email
            result = await send_single_email(
                base_url, user_id, auth_header, 
                email_details['to'], 
                email_details['subject'],
                email_details['body']
            )
            
            sent_emails.append({
                "task": task_key,
                "recipient": email_details['to'],
                "subject": email_details['subject'],
                "status": "sent",
                "message_id": result.get('id')
            })
            
        except Exception as e:
            print(f"Failed to send email for task {task_key}: {e}")
            sent_emails.append({
                "task": task_key,
                "recipient": email_details.get('to', 'unknown') if 'email_details' in locals() else 'unknown',
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "sent_emails": sent_emails,
        "total_sent": len([e for e in sent_emails if e['status'] == 'sent']),
        "total_failed": len([e for e in sent_emails if e['status'] == 'failed']),
        "sent_at": datetime.now(timezone.utc).isoformat()
    }

def extract_email_details(task_data):
    """Extract email details from LLM output - handles multiple formats"""
    
    # Format 1: Direct email object
    if isinstance(task_data, dict) and 'to' in task_data:
        return {
            'to': task_data.get('to'),
            'subject': task_data.get('subject', 'No Subject'),
            'body': task_data.get('body', task_data.get('content', ''))
        }
    
    # Format 2: Object with email details nested
    if isinstance(task_data, dict) and 'email' in task_data:
        email_data = task_data['email']
        return {
            'to': email_data.get('to'),
            'subject': email_data.get('subject', 'No Subject'),
            'body': email_data.get('body', email_data.get('content', ''))
        }
    
    # Format 3: Array of email objects
    if isinstance(task_data, list) and len(task_data) > 0:
        first_email = task_data[0]
        if isinstance(first_email, dict) and 'to' in first_email:
            return {
                'to': first_email.get('to'),
                'subject': first_email.get('subject', 'No Subject'),
                'body': first_email.get('body', first_email.get('content', ''))
            }
    
    # Format 4: Department-based with recipient info
    if isinstance(task_data, dict):
        # Try to find recipient and content in various fields
        recipient = (task_data.get('recipient') or 
                    task_data.get('email') or 
                    task_data.get('to') or 
                    task_data.get('department_email'))
        
        subject = (task_data.get('subject') or 
                  task_data.get('title') or 
                  f"Message from {task_data.get('department', 'System')}")
        
        body = (task_data.get('body') or 
                task_data.get('content') or 
                task_data.get('message') or 
                format_emails_for_department(task_data))
        
        if recipient:
            return {
                'to': recipient,
                'subject': subject,
                'body': body
            }
    
    return None

def format_emails_for_department(task_data):
    """Format email list or content for department emails"""
    content = ""
    
    # If there's an emails array
    if 'emails' in task_data and isinstance(task_data['emails'], list):
        content = f"Department: {task_data.get('department', 'Unknown')}\n\n"
        content += f"Total emails: {len(task_data['emails'])}\n\n"
        
        for i, email in enumerate(task_data['emails'], 1):
            if isinstance(email, dict):
                content += f"{i}. Email ID: {email.get('id', 'N/A')}\n"
                content += f"   Snippet: {email.get('snippet', 'No snippet')}\n\n"
            else:
                content += f"{i}. {email}\n\n"
    
    # If there's a summary or description
    elif 'summary' in task_data:
        content = task_data['summary']
    elif 'description' in task_data:
        content = task_data['description']
    else:
        # Fallback: convert entire object to readable format
        content = str(task_data)
    
    return content

async def send_single_email(base_url, user_id, auth_header, to_email, subject, body):
    """Send a single email using Gmail API"""
    url = f'{base_url}/users/{user_id}/messages/send'
    
    headers = {
        'Authorization': auth_header,
        'Content-Type': 'application/json'
    }
    
    # Create email message
    import base64
    import email.mime.text
    
    message = email.mime.text.MIMEText(body)
    message['to'] = to_email
    message['subject'] = subject
    
    # Convert to base64
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    payload = {
        'raw': raw_message
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()