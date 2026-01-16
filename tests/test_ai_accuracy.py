# import os
# import sys

# # Add project root to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.ai_detector import analyze_text_ai

# def test_ai_detector():
#     print("--- Testing AI Detector Accuracy ---")
    
#     # 1. Known AI (ChatGPT-like formal explanation)
#     ai_text = (
#         "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines "
#         "that are programmed to think like humans and mimic their actions. The term may also "
#         "be applied to any machine that exhibits traits associated with a human mind such "
#         "as learning and problem-solving. AI has become an essential part of the technology "
#         "industry, helping to solve many complex problems in computer science. "
#         "AI is being used to develop and advance numerous fields, including finance, healthcare, "
#         "education, transportation, and more. In healthcare, AI is being used for diverse purposes, "
#         "such as medical diagnosis, drug discovery, and medical research."
#     )
    
#     # 2. Known Human (More conversational, varied sentence structure)
#     human_text = (
#         "I was thinking about how we use computers today, and it's honestly wild. Remember when "
#         "we had to wait for dial-up? Now, everything is instant. I mean, sure, there's a lot of "
#         "talk about robots and stuff, but honestly, I think it's just really fancy math. "
#         "It's helpful for sure, like when I need to find a good recipe for dinner or something, "
#         "but it doesn't feel like it's actually 'thinking'. It's just processing data. "
#         "Anyway, that's just my two cents on the whole situation. What do you think?"
#     )

#     print("\nAnalyzing AI-generated text sample...")
#     ai_score, ai_conclusion = analyze_text_ai(ai_text)
#     print(f"AI Sample Score: {ai_score}%")
#     print(f"Conclusion: {ai_conclusion}")
    
#     print("\nAnalyzing Human-written text sample...")
#     human_score, human_conclusion = analyze_text_ai(human_text)
#     print(f"Human Sample Score: {human_score}%")
#     print(f"Conclusion: {human_conclusion}")

#     print("\n--- RESULTS SUMMARY ---")
#     if ai_score > 60:
#         print("PASS: AI sample correctly identified as Likely/Highly Likely AI.")
#     else:
#         print("FAIL: AI sample score too low.")

#     if human_score < 40:
#         print("PASS: Human sample correctly identified as Human.")
#     else:
#         print("FAIL: Human sample score too high.")

# if __name__ == "__main__":
#     test_ai_detector()
