import sys
import os
import pandas as pd
import random
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plagiarism_detector import detect_plagiarism_from_corpus
from utils.dataset_manager import get_dataset_embeddings

def run_bulk_plagiarism_test(num_samples=100):
    print("====================================================")
    print("      TEXTGUARD PLAGIARISM: BULK ACCURACY TEST      ")
    print("====================================================\n")
    
    # 1. Load Dataset
    print(f"Loading dataset and generating embeddings...")
    start_load = time.time()
    df, _ = get_dataset_embeddings()
    if df is None:
        print("FAIL: Could not load dataset.")
        return
    load_duration = time.time() - start_load
    print(f"Dataset loaded ({len(df)} documents) in {load_duration:.2f}s.\n")

    # 2. Prepare Test Cases
    test_cases = []
    
    # --- POSITIVE CASES (Plagiarized) ---
    # a) Exact snippets from dataset (30 cases)
    print(f"Preparing {num_samples} total test cases...")
    pos_size = num_samples // 2
    neg_size = num_samples - pos_size
    
    # Sample random documents
    samples = df.sample(n=pos_size, random_state=42)
    
    for _, row in samples.iterrows():
        content = row['content']
        title = row['source_title']
        
        # Take a significant snippet (first 3 sentences or first 300 chars)
        snippet = " ".join(content.split()[:50]) # Use first 50 words
        
        test_cases.append({
            "text": snippet,
            "label": "Plagiarized",
            "expected_source": title,
            "category": "Exact Snippet"
        })

    # --- NEGATIVE CASES (Non-Plagiarized) ---
    # Generic human-like text or technical text
    neg_pool = [
        "I was thinking about going to the park later today if the weather stays nice. Maybe bring a book.",
        "To install the dependencies, run npm install in the root directory of your project.",
        "The project management team met yesterday to discuss the roadmap for the next quarter.",
        "Can you please send me the updated invoices by the end of the day? I need them for the report.",
        "The study of economics often involves looking at supply and demand curves to understand market behavior.",
        "I just finished reading a great book about the history of architecture in Japan. Highly recommended.",
        "Does anyone know how to fix a leaking faucet? It's been driving me crazy all night.",
        "The new software update includes several security patches and performance improvements for mobile devices.",
        "We should consider implementing a more robust logging system to catch these errors in production.",
        "It's surprising how much better I feel after getting a full eight hours of sleep for once."
    ]
    
    for i in range(neg_size):
        text = neg_pool[i % len(neg_pool)]
        # Add some variation to avoid exact duplicates in test case set if desired
        if i >= len(neg_pool):
             text += f" (Variation {i})"
             
        test_cases.append({
            "text": text,
            "label": "Original",
            "expected_source": None,
            "category": "Generic Text"
        })

    # 3. Run Tests
    random.shuffle(test_cases)
    
    results = []
    correct_cnt = 0
    total = len(test_cases)
    
    start_test = time.time()
    
    for i, case in enumerate(test_cases):
        print(f"[{i+1}/{total}] Testing {case['category']} ({len(case['text'].split())} words)...")
        
        detection = detect_plagiarism_from_corpus(case['text'])
        score = detection['overall_plagiarism_percentage']
        top_match = detection['top_sources'][0]['source'] if detection['top_sources'] else None
        
        is_correct = False
        if case['label'] == "Plagiarized":
            # Correct if score > 50 and source matches (or at least score is high)
            if score > 50:
                is_correct = True
        else:
            # Correct if score <= 40
            if score <= 40:
                is_correct = True
                
        if is_correct:
            correct_cnt += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        print(f"   Result: Score={score}%, Predicted={'Plagiarized' if score > 50 else 'Original'} | Status: {status}")
        if not is_correct and case['label'] == "Plagiarized":
             print(f"   Note: Expected {case['expected_source']}, Got {top_match}")

    test_duration = time.time() - start_test
    accuracy = (correct_cnt / total) * 100
    
    # 4. Final Report
    print("\n" + "="*50)
    print("                FINAL ACCURACY REPORT               ")
    print("="*50)
    print(f"Total Samples Tested:     {total}")
    print(f"Correct Predictions:      {correct_cnt}")
    print(f"Incorrect Predictions:    {total - correct_cnt}")
    print(f"Overall Accuracy:         {accuracy:.2f}%")
    print(f"Time Taken (Testing):     {test_duration:.2f} seconds")
    print(f"Avg Time Per Sample:      {test_duration/total:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    # The user asked for a 20 test cases for all
    run_bulk_plagiarism_test(num_samples=20)



