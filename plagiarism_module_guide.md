# TextGuard Plagiarism Detection Module

This document outlines the implementation of the plagiarism detection module for the TextGuard project.

## Overview
The module uses **Term Frequency-Inverse Document Frequency (TF-IDF)** and **Cosine Similarity** to detect plagiarism between a source text and a suspicious text. It classifies the similarity into three levels: **Low**, **Moderate**, and **High**.

## Features
- **Preprocessing**: Handles text normalization (lowercase, punctuation removal), tokenization, and stopword removal.
- **Feature Extraction**: Converts text into numerical vectors using `TfidfVectorizer`.
- **Similarity Calculation**: Uses Cosine Similarity to measure the angle between text vectors.
- **Classification**: Categorizes similarity percentage into interpretable levels.
- **No External API Dependencies**: Runs locally using `scikit-learn`.

## Usage
The core logic resides in `utils/plagiarism_detector.py`. You can import the `detect_plagiarism` function directly.

```python
from utils.plagiarism_detector import detect_plagiarism

source = "Original text content..."
suspicious = "Potentially plagiarized content..."

result = detect_plagiarism(source, suspicious)
print(result)
# Output:
# {
#     'similarity_score': 85.5,
#     'plagiarism_level': 'High',
#     'message': 'Significant similarity detected. Likely plagiarism or direct copying.'
# }
```

## Demonstration
A demonstration script `demonstrate_plagiarism.py` is included to showcase the module's capabilities with various test cases:
1. **Direct Copy (High Similarity)**
2. **Paraphrasing (Moderate Similarity)**
3. **Unique Content (Low Similarity)**
4. **Edge Cases (Stopwords Only)**

To run the demo:
```bash
python demonstrate_plagiarism.py
```

## Implementation Details
The module is lightweight and explainable, suitable for academic projects. It avoids deep learning complexities while providing robust results for standard text comparison tasks.
