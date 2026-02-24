import re
import string
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Standard English stopwords (kept for legacy preprocessing if needed)
STOPWORDS = {
    # ... (rest of stopwords)
}

# --- GLOBAL MODEL INSTANCE (Lazy Loaded) ---
_similarity_model = None

def get_similarity_model():
    global _similarity_model
    if _similarity_model is None:
        _similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _similarity_model

def preprocess_text(text):
    """Standardizes text for comparison."""
    if not text: return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    words = text.split()
    return " ".join([w for w in words if w not in STOPWORDS])

def get_plagiarism_level(score):
    """Categorizes similarity scores into risk levels."""
    if score < 10: return "Low", "Little to no similarity detected."
    if score < 30: return "Moderate", "Some similarities found (possible paraphrasing)."
    return "High", "Significant similarity detected (likely plagiarism)."

def calculate_similarity(text1, text2):
    """
    Compares two texts and returns their similarity percentage.
    Uses SentenceTransformer (MiniLM) for Semantic Similarity.
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        model = get_similarity_model()
        
        # 1. Generate embeddings (384-digit math fingerprints)
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        
        # 2. Calculate the "Angle" between meanings (Cosine Similarity)
        cosine_score = util.cos_sim(emb1, emb2)
        
        return round(float(cosine_score) * 100, 2)
        
    except Exception as e:
        print(f"Error in semantic similarity calculation: {e}")
        return 0.0

"""
STUDY NOTE: Legacy Statistical Method (TF-IDF)
---------------------------------------------
This was the original method that looked for exact word overlaps.
It is faster but less "intelligent" than the MiniLM method.

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def calculate_similarity_tfidf(text1, text2):
#     t1_clean = preprocess_text(text1)
#     t2_clean = preprocess_text(text2)
#     vectorizer = TfidfVectorizer(ngram_range=(1, 3))
#     tfidf = vectorizer.fit_transform([t1_clean, t2_clean])
#     sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
#     return round(float(sim[0][0]) * 100, 2)
"""








