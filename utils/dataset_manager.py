import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import torch

# Global instances
_model = None
_dataset = None
_embeddings = None

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dataset.csv')
WIKI_MOVIES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'wiki_movie_plots_deduped.csv')

# How many rows to sample from the large movie dataset (to keep memory manageable)
MOVIE_DATASET_SAMPLE = 10000


def get_model():
    """Lazy load the model."""
    global _model
    if _model is None:
        print("Loading SentenceTransformer model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def reload_dataset():
    """Forces reload of dataset and embeddings."""
    global _dataset, _embeddings
    _dataset = None
    _embeddings = None
    return get_dataset_embeddings()

def _load_and_normalize(path, sample=None):
    """
    Load a CSV and normalize it to a standard format with columns:
    source_id, source_title, content
    """
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1')

    # Detect which schema this CSV uses
    if 'content' in df.columns:
        # Already in our standard format
        df = df[['source_id', 'source_title', 'content']].dropna(subset=['content'])
    elif 'Plot' in df.columns and 'Title' in df.columns:
        # Wiki Movie Plots schema
        df = df.dropna(subset=['Plot'])
        # Limit to a sample to keep embeddings manageable
        if sample and len(df) > sample:
            df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        df = pd.DataFrame({
            'source_id': df.index.astype(str),
            'source_title': df['Title'].astype(str) + ' (' + df['Release Year'].astype(str) + ')',
            'content': df['Plot'].astype(str)
        })
    else:
        print(f"Unknown CSV schema. Columns found: {list(df.columns)}")
        return None

    return df

def get_dataset_embeddings():
    """Loads dataset(s) and generates/caches embeddings."""
    global _dataset, _embeddings

    if _dataset is None:
        combined_frames = []

        # Load standard dataset.csv if it exists
        if os.path.exists(DATASET_PATH):
            print(f"Loading base dataset from {DATASET_PATH}...")
            df_base = _load_and_normalize(DATASET_PATH)
            if df_base is not None:
                combined_frames.append(df_base)

        # Load wiki movie plots dataset if it exists
        if os.path.exists(WIKI_MOVIES_PATH):
            print(f"Loading movie plots dataset from {WIKI_MOVIES_PATH} (sampling {MOVIE_DATASET_SAMPLE} rows)...")
            df_movies = _load_and_normalize(WIKI_MOVIES_PATH, sample=MOVIE_DATASET_SAMPLE)
            if df_movies is not None:
                combined_frames.append(df_movies)
                print(f"Movie plots loaded: {len(df_movies)} entries.")

        if not combined_frames:
            print("No datasets found.")
            return None, None

        _dataset = pd.concat(combined_frames, ignore_index=True)
        print(f"Total dataset size: {len(_dataset)} entries.")

        try:
            model = get_model()
            print("Generating embeddings for dataset (this may take a minute)...")
            content_list = _dataset['content'].tolist()
            _embeddings = model.encode(content_list, convert_to_tensor=True, show_progress_bar=True)
            print("Embeddings generated.")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None, None

    return _dataset, _embeddings

def search_dataset(query_text, top_k=3):
    """
    Searches the dataset for text similar to the query.
    """
    df, dataset_embeddings = get_dataset_embeddings()
    if df is None or dataset_embeddings is None:
        return []

    model = get_model()
    # Skip very short queries
    if len(query_text.split()) < 3:
        return []

    # Create embedding for query
    query_embedding = model.encode(query_text, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(query_embedding, dataset_embeddings)[0]

    # Find top K results
    k = min(top_k, len(df))
    if k == 0:
        return []

    top_results = torch.topk(cosine_scores, k=k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        score = float(score)
        idx = int(idx)

        # Only return relevant matches
        if score < 0.2:
            continue

        results.append({
            'source_title': str(df.iloc[idx].get('source_title', 'Unknown Source')),
            'content': str(df.iloc[idx]['content']),
            'id': str(df.iloc[idx].get('source_id', idx)),
            'score': score
        })

    return results
