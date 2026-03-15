# Plagiarism & Similarity Evaluation Report
Created: 2026-03-13

## 1. Plagiarism Detection (Corpus-Based)
This test evaluates how well the system identifies text copied or paraphrased from the internal database (Photosynthesis, Machine Learning, etc.).

- **Total Tests:** 5
- **Passed:** 3
- **Accuracy:** 60.00%

| Test Case | Expected Source | Found Score | Status | Note |
|:---|:---|:---|:---|:---|
| Exact Match | Machine Learning | 84.64% | **PASS** | Perfect detection for verbatim copies. |
| Strong Paraphrase | Machine Learning | 76.38% | **PASS** | High semantic matching. |
| Weak Paraphrase | Photosynthesis | 81.61% | **PASS** | Successfully identified the core concept. |
| Irrelevant | None | 50.49% | **FAIL** | Score too high for random text (likely semantic noise). |
| Cross-Topic | History/Internet | N/A | **FAIL** | UnboundLocalError fixed; but model confused by overlapping keywords. |

### Observations:
*   **Strengths:** Excellent at semantic retrieval for relevant topics.
*   **Weaknesses:** The "Irrelevant" text check resulted in a 50% score because the semantic model (`MiniLM`) tries to find meaning even in random sentences. 

---

## 2. Text Similarity (Two-Text Comparison)
This test evaluates the accuracy of comparing two arbitrary pieces of text.

- **Total Tests:** 5
- **Passed:** 3
- **Accuracy:** 60.00%

| Test Case | Found Score | Status | Note |
|:---|:---|:---|:---|
| Identical | 100.0% | **PASS** | Perfect match. |
| Capitalization | 92.5% | **FAIL** | Expected >95%, but semantic models vary slightly. |
| Paraphrase | 75.48% | **PASS** | Good semantic understanding. |
| Different Meaning | -0.73% | **FAIL** | Expected 0-40%. Negative score indicates inverse vectors (highly dissimilar). |
| Partial Overlap | 88.99% | **PASS** | Good capture of shared meaning. |

### Observations:
*   The negative score in the "Different Meaning" test is mathematically correct for cosine similarity but should be clamped to 0% for display in the UI.

---

## 3. Bulk Plagiarism Evaluation (100 Samples)
This test evaluates the accuracy of the system across a larger scale batch. A testing script (`tests/plagiarism_bulk_accuracy.py`) dynamically extracted snippets from our datasets and generated unplagiarized text to evaluate the exact thresholds of the sentence-transformers classifier with **100 mixed samples** (50 plagiarized positive instances and 50 original negative instances).

- **Total Tests:** 100
- **Correct Predictions:** 94
- **Incorrect Predictions:** 6
- **Overall Accuracy:** 94.00%

### Observations:
*   With a larger dataset sample size, the semantic matching algorithm's accuracy proves highly reliable.
*   The established risk thresholds correctly separate genuine semantic matches and embeddings matching on random generic noise.

---

## 4. Improvements Implemented
- **Bug Fix:** Fixed an `UnboundLocalError` in `plagiarism_detector.py` where the `level` variable was not defined for low-similarity matches (<50%).
- **Stability:** Added a fallback default `level = 'Low'` to ensure the system never crashes during evaluation.


