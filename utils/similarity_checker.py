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
    # Removes punctuation and standardizes text to lowercase.
    if not text: return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Removes common words (the, is, at) to focus on unique keywords.
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
        
        # Converts both documents into 384-dimensional mathematical vectors.
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        
        # Calculates the final similarity percentage based on the vector angle.
        cosine_score = util.cos_sim(emb1, emb2)
        
        return round(float(cosine_score) * 100, 2)
        
    except Exception as e:
        print(f"Error in semantic similarity calculation: {e}")
        return 0.0













