import re

def chunk_text_test(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Test version of improved chunking with sentence boundary preservation
    """
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed chunk_size, create a new chunk
        if current_word_count + sentence_words > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Keep overlap sentences for context
            overlap_words = 0
            overlap_sentences = []
            for i in range(len(current_chunk) - 1, -1, -1):
                sentence_word_count = len(current_chunk[i].split())
                if overlap_words + sentence_word_count <= overlap:
                    overlap_sentences.insert(0, current_chunk[i])
                    overlap_words += sentence_word_count
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_word_count = overlap_words + sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Test the function
test_text = "This is the first sentence of our test document. This is the second sentence that provides more context. Here we have a third sentence with important information. The fourth sentence continues the narrative flow. This is sentence five which adds even more details. Sentence six provides additional context for testing. The seventh sentence completes our test paragraph. This final sentence wraps up our testing content."

chunks = chunk_text_test(test_text, chunk_size=30, overlap=10)
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk.split())} words):")
    print(f"'{chunk}'")
