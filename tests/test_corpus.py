# import os
# import sys

# # Add project root to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.plagiarism_detector import detect_plagiarism_from_corpus

# def test_corpus_detection():
#     print("--- Testing Corpus-Based Plagiarism Detection ---")
    
#     # Test Case 1: High Plagiarism (Direct Copy from wiki_python.txt)
#     # Using a subset of the text we added to corpus/wiki_python.txt
#     input_text = "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability."
    
#     print(f"\nTest 1: Check expected HIGH plagiarism...")
#     result = detect_plagiarism_from_corpus(input_text)
    
#     print(f"Overall Score: {result['overall_plagiarism_percentage']}%")
#     print(f"Level: {result['plagiarism_level']}")
#     if result['top_sources']:
#         top = result['top_sources'][0]
#         print(f"Top Source: {top['source']} ({top['similarity']}%)")
        
#     assert result['plagiarism_level'] == 'High', "Test 1 Failed: Expected High plagiarism"
#     assert 'wiki_python.txt' in result['top_sources'][0]['source'], "Test 1 Failed: Wrong source identified"
#     print("Test 1 PASSED.")
    
#     # Test Case 2: Unique Content
#     unique_text = "This is a completely unique sentence that I just wrote right now and it should not match anything in the wikipedia corpus ideally."
    
#     print(f"\nTest 2: Check expected LOW plagiarism...")
#     result2 = detect_plagiarism_from_corpus(unique_text)
    
#     print(f"Overall Score: {result2['overall_plagiarism_percentage']}%")
#     print(f"Level: {result2['plagiarism_level']}")
    
#     assert result2['plagiarism_level'] == 'Low', "Test 2 Failed: Expected Low plagiarism"
#     print("Test 2 PASSED.")
    
#     print("\nAll Tests COMPLETED SUCCESSFULLY.")

# if __name__ == "__main__":
#     test_corpus_detection()
