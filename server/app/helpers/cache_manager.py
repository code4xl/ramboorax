import os
import pickle

# Directory to store vector stores
CACHE_DIR = "vector_cache"

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