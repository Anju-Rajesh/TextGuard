import nltk
from utils.dataset_manager import search_dataset

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_sliding_windows(sentences, window_size=3):
    """
    Creates overlapping segments of text (sliding windows) from a list of sentences.
    """
    if len(sentences) <= window_size:
        return [" ".join(sentences)]
    
    windows = []
    for i in range(len(sentences) - window_size + 1):
        window = " ".join(sentences[i : i + window_size])
        windows.append(window)
    return windows

def detect_plagiarism_from_corpus(input_text):
    """
    Detects plagiarism by comparing input text against the dataset using semantic search.
    """
    result = {
        'overall_plagiarism_percentage': 0.0,
        'plagiarism_level': 'Low',
        'top_sources': [],
        'message': ''
    }
    
    if not input_text or not input_text.strip():
        result['message'] = "Input text is empty."
        return result

    # Split input into sentences
    sentences = nltk.sent_tokenize(input_text)
    if not sentences:
        sentences = [input_text]
        
    # 1. Check the entire input text as a whole (Crucial for exact matches)
    windows = [input_text]
    
    # 2. Generate sliding windows for context-aware checking (to catch partial plagiarism)
    windows.extend(get_sliding_windows(sentences, window_size=3))
    
    # 3. Also check individual long sentences to catch direct copy-paste of single sentences
    for s in sentences:
        if len(s.split()) > 8: # Check sentences with > 8 words
            windows.append(s)

            
    unique_sources = {}
    highest_score = 0.0

    # Check each window against the dataset
    # Note: In a production system, we would batch encode these windows for performance
    for window in windows:
        # Search dataset
        matches = search_dataset(window, top_k=1)
        
        for match in matches:
            score = match['score']
            source_title = match['source_title']
            
            # Update highest observed score
            if score > highest_score:
                highest_score = score
            
            # Thresholds for relevance
            if score > 0.3: # 0.3 cosine similarity is a reasonable baseline for semantic match
                if source_title not in unique_sources:
                    unique_sources[source_title] = {
                        'max_score': score,
                        'matched_segments': []
                    }
                else:
                    unique_sources[source_title]['max_score'] = max(unique_sources[source_title]['max_score'], score)
                
                # Store matched segment if it's unique enough (avoid storing same segment multiple times)
                # Simple check: just store it
                unique_sources[source_title]['matched_segments'].append(window)

    # Format output
    final_matches = []
    for title, data in unique_sources.items():
        # Convert cosine similarity (0-1) to percentage (0-100)
        # Cosine similarity for semantic search is usually between 0 and 1.
        # A score of >0.7 is usually very high similarity.
        # We define a mapping:
        # 0.4 -> 40% (Weak)
        # 0.6 -> 60% (Moderate)
        # 0.8 -> 80% (High)
        
        sim_percent = round(data['max_score'] * 100, 2)
        
        level = 'Low'
        if sim_percent > 80:
            level = 'High'
        elif sim_percent > 50:
            level = 'Moderate'
            
        # Get one representative segment
        best_segment = data['matched_segments'][0] if data['matched_segments'] else ""
        
        final_matches.append({
            'source': title,
            'similarity': sim_percent,
            'plagiarism_level': level,
            'matched_segment': best_segment
        })
        
    # Sort by similarity
    final_matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    if final_matches:
        result['overall_plagiarism_percentage'] = final_matches[0]['similarity']
        result['plagiarism_level'] = final_matches[0]['plagiarism_level']
        result['top_sources'] = final_matches
        result['message'] = f"Found matches in {len(final_matches)} source(s)."
    else:
        result['message'] = "No significant matches found in dataset."
        
    return result
