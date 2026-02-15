import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def preprocess_simple(text):
    """
    Standardizes text format: lowercase, remove punctuation, and lemmatize.
    Lemmatization helps match 'running' with 'run', improving paraphrased detection.
    """
    if not text:
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # 3. Tokenize and Lemmatize
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(lemmatized_words)

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
    Refined plagiarism detector using sentence-level and sliding window analysis.
    Now includes lemmatization for better paraphrase detection.
    """
    from utils.corpus_manager import load_corpus
    
    result = {
        'overall_plagiarism_percentage': 0.0,
        'plagiarism_level': 'Low',
        'top_sources': [],
        'message': ''
    }
    
    if not input_text or not input_text.strip():
        result['message'] = "Input text is empty."
        return result
        
    corpus_docs = load_corpus()
    if not corpus_docs:
        result['message'] = "Corpus is empty."
        return result
        
    # Improved Vectorizer settings
    # sublinear_tf=True scales term frequency logarithmically, reducing impact of very frequent words
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        sublinear_tf=True
    )
    
    matches = []
    
    # Preprocess input text (with lemmatization)
    input_clean_for_vec = preprocess_simple(input_text)
    
    # We need raw sentences for display and window logic, but cleaned for TF-IDF
    input_sentences = nltk.sent_tokenize(input_text)
    
    # Prepare input segments
    # We always check individual sentences to catch exact sentence patches
    input_segments_raw = list(input_sentences)
    input_segments_clean = [preprocess_simple(s) for s in input_segments_raw]
    
    # If input is long enough, we ALSO check sliding windows for paragraph context
    if len(input_sentences) > 1:
        # Use sliding window for context
        # Window size 3 is standard, or just the whole text if it's 2-3 sentences
        windows = get_sliding_windows(input_sentences, window_size=3)
        input_segments_raw.extend(windows)
        input_segments_clean.extend([preprocess_simple(s) for s in windows])

    for doc in corpus_docs:
        source_name = doc['source']
        source_content = doc['content']
        
        # Split source into sentences
        source_sentences_raw = nltk.sent_tokenize(source_content)
        if not source_sentences_raw:
            continue
            
        # Preprocess source sentences
        source_sentences_clean = [preprocess_simple(s) for s in source_sentences_raw]
        
        # We also create source windows for paragraph matching
        source_windows_raw = []
        source_windows_clean = []
        if len(source_sentences_raw) > 3:
            source_windows_raw = get_sliding_windows(source_sentences_raw, window_size=3)
            source_windows_clean = [preprocess_simple(s) for s in source_windows_raw]

        # Combine all for vectorization fitting
        # We need a common vocabulary
        all_corpus_clean = source_sentences_clean + source_windows_clean
        
        try:
            # We fit on EVERYTHING to ensure valid dimensions
            # But we only care about cosine sim between input and this doc's segments
            
            # Optimization: Fit on input + this doc only
            # This is faster than fitting on entire corpus every time
            vocab_data = input_segments_clean + all_corpus_clean
            tfidf_matrix = vectorizer.fit_transform(vocab_data)
            
            # Indices
            input_len = len(input_segments_clean)
            
            # Vectors
            input_vecs = tfidf_matrix[0:input_len]
            corpus_vecs = tfidf_matrix[input_len:]
            
            # Calculate Similarity
            # We want best match for ANY input segment against ANY corpus segment
            sim_matrix = cosine_similarity(input_vecs, corpus_vecs)
            
            # Find max
            max_val = sim_matrix.max()
            
            # To report the matched segment, we need the index in corpus_vecs
            # flat argmax
            flat_idx = sim_matrix.argmax()
            # row, col
            row_idx, col_idx = divmod(flat_idx, sim_matrix.shape[1])
            
            max_sim_score = sim_matrix[row_idx, col_idx]
            
            # Map col_idx back to raw text
            # vocab_data construction was: input (0..N) + sentences + windows
            # corpus_vecs starts at 0 relative to itself
            
            if col_idx < len(source_sentences_raw):
                matched_text = source_sentences_raw[col_idx]
            else:
                # It was a window match
                win_idx = col_idx - len(source_sentences_raw)
                matched_text = source_windows_raw[win_idx]
            
            percentage = round(float(max_sim_score) * 100, 2)
            
            level = "Low"
            # Adjusted thresholds as requested to be more sensitive
            if percentage > 50:
                level = "High"
            elif percentage > 25:
                level = "Moderate"
                
            if percentage > 10: # Lowered reporting threshold
                matches.append({
                    'source': source_name,
                    'similarity': percentage,
                    'plagiarism_level': level,
                    'matched_segment': matched_text
                })

        except Exception as e:
            # In case of empty vocabulary or other scikit errors
            continue

    # Sort and finalize
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    if matches:
        top_score = matches[0]['similarity']
        result['overall_plagiarism_percentage'] = top_score
        result['top_sources'] = matches
        result['plagiarism_level'] = matches[0]['plagiarism_level']
        result['message'] = f"Found matches in {len(matches)} source(s)."
    else:
        result['message'] = "No significant matches found."
            
    return result
