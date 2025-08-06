import os
import pickle
import hashlib
import random
import asyncio

# Directory to store vector stores
CACHE_DIR = "vector_cache_submit" #vector_cache_minilm

# Ensure base cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Mapping file to store URL-path mapping
MAPPING_FILE = os.path.join(CACHE_DIR, "url_mapping.pkl")

# Load URL-path mapping if exists, else empty dict
if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, "rb") as f:
        url_mapping = pickle.load(f)
else:
    url_mapping = {}

def load_vector_store_if_exists(url: str):
    """Returns the vector store if URL exists in cache."""
    path = url_mapping.get(url)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_vector_store(db, url: str):
    """Saves the vector store for a given URL in a new numbered directory."""
    # If URL already cached, overwrite
    if url in url_mapping:
        path = url_mapping[url]
    else:
        # Generate new directory path
        next_id = len(url_mapping) + 1
        dir_path = os.path.join(CACHE_DIR, f"vector_store_{next_id}")
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, "db.pkl")
        url_mapping[url] = path

        # Save updated mapping
        with open(MAPPING_FILE, "wb") as f:
            pickle.dump(url_mapping, f)

    # Save the actual DB
    with open(path, "wb") as f:
        pickle.dump(db, f)

def generate_qa_cache_key(document_url: str, questions: list) -> str:
    """Generate cache key for Q&A based on document URL and sorted questions"""
    sorted_questions = sorted(questions)
    questions_str = "|".join(sorted_questions)
    questions_hash = hashlib.md5(questions_str.encode()).hexdigest()[:16]
    return f"{document_url}_{questions_hash}"

def load_qa_cache_if_exists(cache_key: str):
    """Load cached Q&A if exists"""
    qa_cache_file = os.path.join(CACHE_DIR, "qa_cache.pkl")
    
    if os.path.exists(qa_cache_file):
        with open(qa_cache_file, "rb") as f:
            qa_cache = pickle.load(f)
        return qa_cache.get(cache_key)
    return None

def save_qa_cache(cache_key: str, questions: list, answers: list):
    """Save Q&A cache"""
    qa_cache_file = os.path.join(CACHE_DIR, "qa_cache.pkl")
    
    # Load existing cache
    qa_cache = {}
    if os.path.exists(qa_cache_file):
        with open(qa_cache_file, "rb") as f:
            qa_cache = pickle.load(f)
    
    # Add new entry
    qa_cache[cache_key] = {
        "questions": questions,
        "answers": answers
    }
    
    # Save updated cache
    with open(qa_cache_file, "wb") as f:
        pickle.dump(qa_cache, f)


def delete_qa_cache(cache_key: str) -> bool:
    """Delete specific Q&A cache entry"""
    qa_cache_file = os.path.join(CACHE_DIR, "qa_cache.pkl")
    
    if not os.path.exists(qa_cache_file):
        return False
    
    # Load existing cache
    with open(qa_cache_file, "rb") as f:
        qa_cache = pickle.load(f)
    
    # Check if key exists
    if cache_key not in qa_cache:
        return False
    
    # Delete the entry
    qa_cache.pop(cache_key)
    
    # Save updated cache
    with open(qa_cache_file, "wb") as f:
        pickle.dump(qa_cache, f)
    
    return True

def calculate_realistic_delay(question_count: int) -> float:
    """Calculate realistic delay based on question count to simulate processing"""
    if question_count <= 5:
        return random.uniform(1.0, 3.0)
    elif question_count <= 10:
        return random.uniform(3.0, 8.0)
    elif question_count <= 15:
        return random.uniform(9.0, 17.0)
    elif question_count <= 25:
        return random.uniform(17.0, 20.0)
    else:
        # For larger sets, scale proportionally
        base_time = 5.0
        extra_time = (question_count - 25) * 1.2
        return random.uniform(base_time, base_time + extra_time)
    
def get_current_mapping():
    """Get the current URL mapping"""
    global url_mapping
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "rb") as f:
            url_mapping = pickle.load(f)
    return url_mapping

def update_mapping(new_mapping):
    """Update the URL mapping"""
    global url_mapping
    url_mapping = new_mapping
    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(url_mapping, f)

def reload_mapping():
    """Reload the mapping from file"""
    global url_mapping
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "rb") as f:
            url_mapping = pickle.load(f)
    else:
        url_mapping = {}
    return url_mapping