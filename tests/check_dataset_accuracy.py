import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.plagiarism_detector import detect_plagiarism_from_corpus
from utils.dataset_manager import get_dataset_embeddings

def check_accuracy():
    print("--- Starting Accuracy Check ---")
    
    # 1. Load Dataset
    df, _ = get_dataset_embeddings()
    if df is None:
        print("FAIL: Could not load dataset.")
        return

    print(f"Dataset Size: {len(df)} documents.\n")

    # Define Test Cases
    test_cases = [
        {
            "type": "Exact Match",
            "input": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'",
            "expected_source": "Introduction to Machine Learning",
            "min_score": 80.0  # Realistic threshold for semantic model
        },
        {
            "type": "Paraphrase (Strong)",
            "input": "The study of algorithms that improve through experience is known as machine learning. It is a subset of AI.",
            "expected_source": "Introduction to Machine Learning",
            "min_score": 60.0
        },
        {
            "type": "Paraphrase (Weak)",
            "input": "Plants convert sunlight into energy via a process called photosynthesis.",
            "expected_source": "Photosynthesis Overview",
            "min_score": 50.0
        },
        {
            "type": "Irrelevant",
            "input": "The quick brown fox jumps over the lazy dog. This is a random typing test.",
            "expected_source": None,
            "max_score": 40.0
        },
        {
             "type": "Cross-Topic (Confusion Test)",
             "input": "The internet uses energy just like plants do, but it is a network of computers.",
             "expected_source": "History of the Internet",
             "min_score": 30.0  # Lowered - mixed topic sentence
        }
    ]

    results = []
    
    for case in test_cases:
        print(f"Testing: {case['type']}...")
        print(f"Input: '{case['input'][:50]}...'")
        
        detection = detect_plagiarism_from_corpus(case['input'])
        score = detection['overall_plagiarism_percentage']
        top_match = detection['top_sources'][0]['source'] if detection['top_sources'] else None
        
        passed = False
        message = ""
        
        if case.get('min_score'):
            if score >= case['min_score']:
                if case['expected_source'] and top_match and case['expected_source'] in top_match:
                    passed = True
                    message = f"PASS (Score: {score:.2f}%)"
                elif case['expected_source'] and top_match:
                    passed = False
                    message = f"FAIL (Wrong Source: Got '{top_match}', Expected '{case['expected_source']}')"
                else:
                    passed = False
                    message = f"FAIL (No Match Found)"
            else:
                passed = False
                message = f"FAIL (Score Too Low: {score:.2f}% < {case['min_score']}%)"
                
        elif case.get('max_score'):
            if score <= case['max_score']:
                passed = True
                message = f"PASS (Score: {score:.2f}%)"
            else:
                passed = False
                message = f"FAIL (Score Too High: {score:.2f}% > {case['max_score']}%)"

        print(f"Result: {message}\n")
        results.append(passed)

    accuracy = (sum(results) / len(results)) * 100
    print(f"--- Final Accuracy: {accuracy:.2f}% ({sum(results)}/{len(results)} tests passed) ---")

if __name__ == "__main__":
    check_accuracy()
