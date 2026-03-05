import sys
import os


# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.similarity_checker import calculate_similarity

def check_similarity_accuracy():
    print("--- Starting Similarity Accuracy Check (20 Cases) ---")
    
    test_cases = [
        # 1-5: Existing base
        {"type": "Identical", "text1": "The quick brown fox jumps.", "text2": "The quick brown fox jumps.", "expected_range": (99.0, 100.0)},
        {"type": "Capitalization", "text1": "Hello World!", "text2": "hello world", "expected_range": (90.0, 100.0)},
        {"type": "Synonyms", "text1": "Recent studies show that climate change is accelerating.", "text2": "New research indicates that global warming is happening faster.", "expected_range": (70.0, 100.0)},
        {"type": "Different Meaning", "text1": "I love eating fresh apples in the morning.", "text2": "The stock market crashed after the election.", "expected_range": (-50.0, 40.0)}, # some cosine similarities go negative
        {"type": "Partial Overlap", "text1": "Artificial Intelligence is the future of technology.", "text2": "Technology will be shaped by Artificial Intelligence.", "expected_range": (70.0, 100.0)},
        
        # 6-10: Variations and additions
        {"type": "Antonyms", "text1": "I absolutely love the hot summer weather.", "text2": "I absolutely hate the freezing winter weather.", "expected_range": (0.0, 60.0)},
        {"type": "Reordered", "text1": "The quick brown fox jumps over the lazy dog.", "text2": "The lazy dog was jumped over by the quick brown fox.", "expected_range": (80.0, 100.0)},
        {"type": "Extra words", "text1": "The cat slept.", "text2": "The very lazy old black cat slept soundly on the rug.", "expected_range": (50.0, 95.0)},
        {"type": "Different topics", "text1": "Learning to program in Python is fun.", "text2": "Playing the acoustic guitar requires practice.", "expected_range": (-50.0, 40.0)},
        {"type": "Same topic, different fact", "text1": "Albert Einstein was born in 1879.", "text2": "Isaac Newton was born in 1642.", "expected_range": (20.0, 70.0)},
        
        # 11-15: Grammatical & semantic nuances
        {"type": "Opposite sentiment", "text1": "I am very happy today.", "text2": "I am very sad today.", "expected_range": (30.0, 80.0)}, # Contextually similar but opposite meaning
        {"type": "Pluralization", "text1": "A cat is a cute pet.", "text2": "Cats are cute pets.", "expected_range": (85.0, 100.0)},
        {"type": "Idiomatic vs Literal", "text1": "He spilled the beans about the surprise party.", "text2": "He revealed the secret about the surprise party.", "expected_range": (40.0, 90.0)},
        {"type": "Numbers", "text1": "I have 2 green apples.", "text2": "I have two green apples.", "expected_range": (90.0, 100.0)},
        {"type": "Typo / Misspelling", "text1": "This is a simple test.", "text2": "Tihs is a simpel tast.", "expected_range": (70.0, 100.0)},
        
        # 16-20: Advanced semantic
        {"type": "Contractions", "text1": "Do not go into the dark forest.", "text2": "Don't go into the dark forest.", "expected_range": (95.0, 100.0)},
        {"type": "Active vs Passive", "text1": "The dog bit the mailman.", "text2": "The mailman was bitten by the dog.", "expected_range": (85.0, 100.0)},
        {"type": "Translated Idiom", "text1": "It is raining cats and dogs.", "text2": "It is pouring heavy rain outside.", "expected_range": (30.0, 90.0)},
        {"type": "Completely Empty vs Text", "text1": "", "text2": "Some random text to compare against.", "expected_range": (0.0, 0.0) }, # Depending on implementation could be 0
        {"type": "Same Characters", "text1": "A B C D E F G", "text2": "A B C D E F G", "expected_range": (99.0, 100.0)}
    ]

    results = []
    
    for i, case in enumerate(test_cases):
        # Handle empty string gracefully by returning 0 if implementation fails
        try:
            score = calculate_similarity(case['text1'], case['text2'])
        except Exception:
            score = 0.0
            
        passed = case['expected_range'][0] <= score <= case['expected_range'][1]
        status = "PASS" if passed else "FAIL"
        print(f"[{i+1}/20 - {case['type']}] Score: {score:.2f}% | Status: {status}")
        if not passed:
             print(f"    Expected between {case['expected_range'][0]}% and {case['expected_range'][1]}%")
        results.append(passed)

    accuracy = (sum(results) / len(results)) * 100
    print(f"\n--- Final Accuracy: {accuracy:.2f}% ({sum(results)}/{len(results)} tests passed) ---")

if __name__ == "__main__":
    check_similarity_accuracy()
