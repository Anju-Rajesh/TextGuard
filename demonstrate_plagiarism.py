from utils.plagiarism_detector import detect_plagiarism
import time

def print_result(case_name, source, suspicious, result):
    print(f"\n--- {case_name} ---")
    print(f"Source Text:     {source[:60]}..." if len(source) > 60 else f"Source Text:     {source}")
    print(f"Suspicious Text: {suspicious[:60]}..." if len(suspicious) > 60 else f"Suspicious Text: {suspicious}")
    print("-" * 30)
    print(f"Similarity Score: {result['similarity_score']}%")
    print(f"Plagiarism Level: {result['plagiarism_level']}")
    print(f"Analysis Message: {result['message']}")
    print("=" * 60)

def main():
    print("================================================")
    print("   TextGuard Plagiarism Detection Module Demo   ")
    print("================================================\n")
    
    # Case 1: High Similarity (Direct Copy)
    source_1 = "Natural Language Processing is a subfield of linguistics, computer science, and artificial intelligence."
    suspicious_1 = "Natural Language Processing is a subfield of linguistics, computer science, and artificial intelligence."
    result_1 = detect_plagiarism(source_1, suspicious_1)
    print_result("Test Case 1: Direct Copy (High Similarity)", source_1, suspicious_1, result_1)
    
    # Case 2: Moderate Similarity (Paraphrasing)
    source_2 = "Machine learning algorithms build a model based on sample data, known as training data."
    suspicious_2 = "Algorithms in machine learning build models using sample data called training data."
    result_2 = detect_plagiarism(source_2, suspicious_2)
    print_result("Test Case 2: Paraphrasing (Moderate Similarity)", source_2, suspicious_2, result_2)

    # Case 3: Low Similarity (Different Topics)
    source_3 = "The photosynthesis process is crucial for plant survival and helps produce oxygen."
    suspicious_3 = "Stock markets are volatile and subject to various economic factors globally."
    result_3 = detect_plagiarism(source_3, suspicious_3)
    print_result("Test Case 3: Unique Content (Low Similarity)", source_3, suspicious_3, result_3)

    # Case 4: Edge Case (Stopwords only)
    source_4 = "The and or but if"
    suspicious_4 = "It they we are"
    result_4 = detect_plagiarism(source_4, suspicious_4)
    print_result("Test Case 4: Stopwords Only (Edge Case)", source_4, suspicious_4, result_4)

    print("\nDemonstration Completed Successfully.")

if __name__ == "__main__":
    main()
