import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Standard English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}



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
    Uses preprocessing, TF-IDF, and Cosine Similarity.
    """
    # 1. Validation and Preprocessing
    t1_clean = preprocess_text(text1)
    t2_clean = preprocess_text(text2)
    
    if not t1_clean or not t2_clean:
        return 0.0
    
    # 2. Vectorization
    # We use n-grams (1-3) to capture phrases, which is better for detecting partial plagiarism.
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    
    try:
        tfidf = vectorizer.fit_transform([t1_clean, t2_clean])
        
        # 3. Cosine Similarity
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
        return round(float(sim[0][0]) * 100, 2)
        
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return 0.0








