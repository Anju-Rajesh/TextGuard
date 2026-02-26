import os
import sys
import time

# Add project root to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai_detector import analyze_text_ai

def run_bulk_test():
    """
    Runs a batch of AI and Human samples through the detector to calculate overall accuracy.
    """
    # --- TEST DATASET (Exactly 20 cases) ---
    
    test_data = [
        # AI SAMPLES (10 Known AI-generated)
        {"text": "The industrial revolution was a pivotal period in human history that began in the late 18th century. It marked a transition from agricultural societies to industrial ones, characterized by the introduction of machinery and mass production techniques. Steam power played a critical role in this transformation, enabling the development of factories and improved transportation systems like railways. As a result, urbanization increased rapidly as people moved from rural areas to cities in search of employment opportunities in the newly established industries.", "label": "AI"},
        {"text": "Climate change remains one of the most pressing global challenges of our time. Scientific evidence clearly indicates that human activities, particularly the burning of fossil fuels and deforestation, have significantly contributed to the rise in global temperatures. This warming trend has led to the melting of polar ice caps, rising sea levels, and an increase in the frequency of extreme weather events. International agreements like the Paris Agreement aim to mitigate these effects by encouraging nations to reduce their carbon footprints and transition to renewable energy sources.", "label": "AI"},
        {"text": "In the realm of modern medicine, gene editing technologies such as CRISPR-Cas9 have opened up unprecedented possibilities for treating genetic disorders. By allowing scientists to precisely modify DNA sequences, these tools offer the potential to cure diseases that were previously thought to be incurable. However, the application of such technology also raises significant ethical concerns regarding the potential for 'designer babies' and the long-term impact on the human gene pool. Continued dialogue between scientists, ethicists, and policymakers is essential to ensure responsible use.", "label": "AI"},
        {"text": "Space exploration has transitioned from a competitive race between superpowers to a collaborative international effort. The International Space Station (ISS) stands as a testament to what humanity can achieve when nations work together toward a common goal. Looking forward, the focus has shifted toward deep space exploration, with missions aimed at returning to the moon and eventually reaching Mars. Commercial aerospace companies are also playing an increasingly vital role, reducing costs and accelerating the pace of innovation in rocket technology and satellite deployment.", "label": "AI"},
        {"text": "Artificial Intelligence is rapidly reshaping the landscape of modern industries, from healthcare to finance. In medicine, AI-driven diagnostic tools are aiding doctors in identifying diseases with greater precision and speed than ever before. In the financial sector, algorithms are used to detect fraudulent transactions and optimize investment strategies in real-time. However, the rise of AI also brings forth complex ethical considerations, particularly regarding data privacy, algorithmic bias, and the potential displacement of jobs.", "label": "AI"},
        {"text": "The transition to renewable energy is a critical component of global efforts to combat climate change. Wind, solar, and hydroelectric power are becoming increasingly cost-competitive with traditional fossil fuels, leading to a significant increase in their adoption worldwide. Technological advancements in energy storage, such as high-capacity lithium-ion batteries, are addressing the intermittency issues associated with renewable sources. Moreover, many countries are implementing policies and subsidies to encourage the development of green infrastructure.", "label": "AI"},
        {"text": "Machine learning is a subset of artificial intelligence that involves the use of algorithms and statistical models to enable computers to perform a specific task without being explicitly programmed. It relies heavily on large datasets to train models, which use to make predictions or decisions based on new data. Common applications include email filtering, computer vision, and predictive text analysis.", "label": "AI"},
        {"text": "Blockchain technology provides a decentralized, secure, and transparent way to record transactions across multiple computers. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data, making it inherently resistant to modification. This makes it an ideal underlying protocol for cryptocurrencies like Bitcoin and Ethereum, as well as supply chain tracking and secure voting systems.", "label": "AI"},
        {"text": "The psychological concept of cognitive dissonance explains the mental discomfort experienced by individuals who hold two or more contradictory beliefs, ideas, or values. This theory, first proposed by Leon Festinger in 1957, suggests that humans strive for internal consistency. When dissonance occurs, individuals may alter their attitudes, beliefs, or actions to reduce the psychological stress and restore harmony.", "label": "AI"},
        {"text": "Quantum computing represents a fundamental shift in how complex problems are solved computationally. Unlike classical bits that can only be 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously due to superposition and entanglement. This allows quantum computers to process massive amounts of data exponentially faster than traditional supercomputers, with profound implications for cryptography and material science.", "label": "AI"},
        
        # HUMAN SAMPLES (10 Known Human-written/Conversational)
        {"text": "So I went to this new coffee shop down the street yesterday morning. Honestly, the vibe was great but the espresso was way too acidic for my taste. I usually like a darker roast but they only had this light citrusy blend available. It's funny how everyone is obsessed with 'third wave' coffee these days when sometimes you just want a simple cup of Joe that doesn't taste like a lemon. Anyway, I think I'll stick to my home brew for a while unless I'm just looking for a cool place to sit and read.", "label": "Human"},
        {"text": "Programming is one of those things that makes you feel like a genius one minute and a complete idiot the next. I spent four hours today chasing a bug that turned out to be a missing semicolon in a config file. It's frustrating, sure, but that moment when the code finally runs and everything just clicks? That's the best feeling in the world. It's like solving a puzzle that keeps changing its rules while you're playing. I guess that's why people get hooked on it despite the headaches.", "label": "Human"},
        {"text": "Honestly, I don't get the hype about that new sci-fi movie everyone is talking about. The visual effects were stunning, don't get me wrong, but the plot felt so hollow. It was just a bunch of tropes stitched together with no real heart behind the characters. I found myself checking my watch halfway through because I just couldn't bring myself to care what happened to the 'chosen one' this time around. I miss movies that focused more on the writing than just trying to blow stuff up every five minutes.", "label": "Human"},
        {"text": "I spent the entire afternoon at the old library downtown today. You know, the one with the huge oak doors that creak every time someone walks in? I wasn't even looking for anything specific, just wanted to escape the rain for a bit. I found this stack of old travel journals from the 1920s in the basement section. Most of the pages were yellowed and brittle, but the drawings were incredible. Just raw, hand-sketched scenes from places that probably don't even look like that anymore.", "label": "Human"},
        {"text": "My dog, Buster, has this weird habit of bringing me a single sock whenever I come home from work. It's never a pair, and it's always one from the bottom of the laundry basket. He doesn't chew it or anything, he just holds it in his mouth and wiggles his whole body until I acknowledge him. It's honestly the highlight of my day. I have no idea why he started doing it, but it’s become our little ritual. Life would be so much quieter without him, but I wouldn't trade that goofy greeting for anything.", "label": "Human"},
        {"text": "Look, I know everyone says sourdough is the ultimate quarantine hobby, but I think I've finally mastered it. My first three loaves were basically expensive doorstops—hard as rock and totally flat. But this morning? Total game changer. The crust actually had that crackle when I tapped it, and the inside was all airy and perfect. I followed this random recipe I found on a forum from like 2012 that suggested adding a tiny bit of rye flour. Turns out that was the secret sauce.", "label": "Human"},
        {"text": "Traffic on I-95 this morning was an absolute nightmare. I've been doing this commute for three years and I don't think I've ever seen it backed up all the way to exit 4. Some guy in a pickup truck completely cut me off trying to get to the off-ramp at the last second. It's a miracle there wasn't an accident. I ended up being twenty minutes late for my morning meeting, which my boss wasn't exactly thrilled about. Sometimes I seriously consider moving closer to the office.", "label": "Human"},
        {"text": "We just got back from our trip to the mountains and I'm already exhausted. We hiked this trail that was supposedly 'moderate' but it felt basically vertical the entire way up. The view at the top was completely worth it though; you could see the entire valley and lake below. We brought some sandwiches from a deli back in town and just sat on a rock for an hour eating and taking it all in. I just wish I brought better boots because my ankles are killing me.", "label": "Human"},
        {"text": "If you want to make a really good grilled cheese, you have to use mayo on the outside of the bread instead of butter. I know it sounds super weird, but my grandma taught me this trick years ago. It cooks up so much crispier and doesn't burn as easily in the pan. Plus, you throw a little bit of garlic powder in the mayo before you spread it? Incredible. Seriously, try it next time you're craving one and tell me it isn't better.", "label": "Human"},
        {"text": "I was looking through some old boxes in my garage today and found my old GameBoy Color from when I was a kid. The crazy part is it still had batteries in it and Pokemon Red was still wedged in the back. I flipped the switch and the screen actually turned on! The little startup noise hit me with a wave of nostalgia so hard I actually had to sit down. I ended up playing for like an hour before I realized I was supposed to be cleaning.", "label": "Human"}
    ]

    print("====================================================")
    print("      TEXTGUARD AI: BULK ACCURACY TESTER (20)      ")
    print("====================================================\n")
    print(f"Loading {len(test_data)} samples for evaluation...\n")

    correct_cnt = 0
    total_cnt = len(test_data)
    
    start_time = time.time()

    for i, item in enumerate(test_data):
        text = item["text"]
        label = item["label"]
        
        print(f"[{i+1}/{total_cnt}] Testing {label} sample ({len(text.split())} words)...")
        
        # Run detection
        score, conclusion = analyze_text_ai(text)
        
        # Determine prediction success
        # We consider a prediction correct if AI score > 50 for AI label, or <= 50 for Human label
        is_ai_prediction = score > 50
        is_correct = (is_ai_prediction and label == "AI") or (not is_ai_prediction and label == "Human")
        
        if is_correct:
            correct_cnt += 1
            status = "CORRECT"
        else:
            status = "FAILED"
            
        print(f"   Result: Score={score}%, Predicted={'AI' if is_ai_prediction else 'Human'} | Status: {status}")
        print("-" * 50)

    end_time = time.time()
    accuracy = (correct_cnt / total_cnt) * 100
    duration = end_time - start_time

    print("\n" + "="*50)
    print("                FINAL ACCURACY REPORT               ")
    print("="*50)
    print(f"Total Samples Tested:     {total_cnt}")
    print(f"Correct Predictions:      {correct_cnt}")
    print(f"Incorrect Predictions:    {total_cnt - correct_cnt}")
    print(f"Overall Accuracy:         {accuracy:.2f}%")
    print(f"Time Taken:               {duration:.2f} seconds")
    print("="*50)
    
    if accuracy >= 80:
        print("PERFORMANCE: EXCELLENT")
    elif accuracy >= 60:
        print("PERFORMANCE: GOOD (Needs Calibration)")
    else:
        print("PERFORMANCE: WEAK (Check Model Settings)")
    print("="*50)

if __name__ == "__main__":
    run_bulk_test()
