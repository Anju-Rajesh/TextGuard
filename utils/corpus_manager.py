import os

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corpus')

def load_corpus():
    """
    Loads all text files from the corpus directory.
    
    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - 'source': Filename of the document.
              - 'content': The text content of the document.
    """
    documents = []
    
    # Create corpus directory if it doesn't exist (safety check)
    if not os.path.exists(CORPUS_DIR):
        os.makedirs(CORPUS_DIR)
        return documents

    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(CORPUS_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip(): # Only add non-empty files
                        documents.append({
                            'source': filename,
                            'content': content
                        })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    return documents
