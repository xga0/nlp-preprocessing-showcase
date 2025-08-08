#!/usr/bin/env python3
"""
Demo script to demonstrate the three packages: lightlemma, emoticon_fix, and contraction_fix.
This script shows how these packages work together to preprocess text for sentiment analysis.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_packages():
    """Demonstrate the three packages with example texts."""
    
    print("="*80)
    print("DEMONSTRATING LIGHTLEMMA, EMOTICON_FIX, AND CONTRACTION_FIX")
    print("="*80)
    
    # Example texts that showcase the packages
    example_texts = [
        "I don't like this product at all! 😡 It's terrible and I'm not happy with it.",
        "I love this new iPhone! It's amazing and works perfectly! 😍",
        "This service is okay, nothing special. 🤷‍♂️",
        "Can't believe how bad this is! 😠 Don't buy it!",
        "The weather is nice today, I'm going for a walk! 😊",
        "This movie was decent, not great but not bad either.",
        "I don't care about this topic at all, it's irrelevant to me.",
        "This restaurant has the best food ever! Highly recommend! ⭐⭐⭐⭐⭐"
    ]
    
    try:
        # Import the packages
        from lightlemma import text_to_lemmas, text_to_stems
        from emoticon_fix import replace_emoticons as fix_emoticons
        from contraction_fix import fix as fix_contractions
        
        print("SUCCESS: All packages imported successfully!")
        print()
        
        for i, text in enumerate(example_texts, 1):
            print(f"Example {i}:")
            print(f"Original: {text}")
            print("-" * 50)
            
            # Step 1: Fix contractions
            contracted_text = fix_contractions(text)
            print(f"After contraction_fix: {contracted_text}")
            
            # Step 2: Fix emoticons
            emoticon_fixed_text = fix_emoticons(contracted_text)
            print(f"After emoticon_fix: {emoticon_fixed_text}")
            
            # Step 3: Show lemmatization
            lemmas = text_to_lemmas(emoticon_fixed_text)
            print(f"After lightlemma (lemmatization): {' '.join(lemmas)}")
            
            # Step 4: Show stemming
            stems = text_to_stems(emoticon_fixed_text)
            print(f"After lightlemma (stemming): {' '.join(stems)}")
            
            print("="*80)
            print()
    
    except ImportError as e:
        print(f"FAILED: Error importing packages: {e}")
        print("Please install the required packages:")
        print("pip install lightlemma emoticon-fix contraction-fix")
        return False
    
    return True


def demo_individual_packages():
    """Demonstrate each package individually with more examples."""
    
    print("\n" + "="*80)
    print("INDIVIDUAL PACKAGE DEMONSTRATIONS")
    print("="*80)
    
    try:
        from lightlemma import text_to_lemmas, text_to_stems
        from emoticon_fix import replace_emoticons as fix_emoticons
        from contraction_fix import fix as fix_contractions
        
        # Contraction examples
        print("\nCONTRACTION_FIX EXAMPLES:")
        print("-" * 40)
        contraction_examples = [
            "I don't like this",
            "You can't do that",
            "It's a great day",
            "I'm going home",
            "We'll see you there",
            "They've been working hard",
            "She'd like to come",
            "He won't be there"
        ]
        
        for text in contraction_examples:
            fixed = fix_contractions(text)
            print(f"{text} → {fixed}")
        
        # Emoticon examples
        print("\nEMOTICON_FIX EXAMPLES:")
        print("-" * 40)
        emoticon_examples = [
            "I'm happy! 😊",
            "This is terrible! 😡",
            "I love it! 😍",
            "I'm sad 😢",
            "This is funny 😂",
            "I'm confused 🤔",
            "Great job! 👏",
            "I'm tired 😴"
        ]
        
        for text in emoticon_examples:
            fixed = fix_emoticons(text)
            print(f"{text} → {fixed}")
        
        # Lemmatization examples
        print("\nLIGHTLEMMA EXAMPLES:")
        print("-" * 40)
        lemmatization_examples = [
            "The cats are running faster than dogs",
            "I am studying for my exams",
            "She has been working hard",
            "They will be coming tomorrow",
            "The books are on the shelves"
        ]
        
        for text in lemmatization_examples:
            lemmas = text_to_lemmas(text)
            stems = text_to_stems(text)
            print(f"Original: {text}")
            print(f"Lemmas: {' '.join(lemmas)}")
            print(f"Stems:  {' '.join(stems)}")
            print()
    
    except ImportError as e:
        print(f"FAILED: Error: {e}")
        return False
    
    return True


def demo_performance_comparison():
    """Demonstrate performance benefits of the packages."""
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    try:
        import time
        from lightlemma import text_to_lemmas
        from emoticon_fix import replace_emoticons as fix_emoticons
        from contraction_fix import fix as fix_contractions
        
        # Test text
        test_text = "I don't like this product! 😡 It's terrible and I'm not happy with it. Can't believe how bad this is! 😠 Don't buy it!"
        
        # Test contraction_fix performance
        start_time = time.time()
        for _ in range(1000):
            fix_contractions(test_text)
        contraction_time = time.time() - start_time
        
        # Test emoticon_fix performance
        start_time = time.time()
        for _ in range(1000):
            fix_emoticons(test_text)
        emoticon_time = time.time() - start_time
        
        # Test lightlemma performance
        start_time = time.time()
        for _ in range(1000):
            text_to_lemmas(test_text)
        lemmatization_time = time.time() - start_time
        
        print(f"Processing 1000 texts:")
        print(f"contraction_fix: {contraction_time:.4f} seconds")
        print(f"emoticon_fix:   {emoticon_time:.4f} seconds")
        print(f"lightlemma:     {lemmatization_time:.4f} seconds")
        print(f"Total time:     {contraction_time + emoticon_time + lemmatization_time:.4f} seconds")
        
        # Calculate throughput
        throughput = 1000 / (contraction_time + emoticon_time + lemmatization_time)
        print(f"Throughput:     {throughput:.0f} texts/second")
        
    except ImportError as e:
        print(f"FAILED: Error: {e}")
        return False
    
    return True


def main():
    """Main demo function."""
    
    print("Starting package demonstrations...")
    print()
    
    # Demo 1: Combined package demonstration
    success1 = demo_packages()
    
    if success1:
        # Demo 2: Individual package demonstrations
        success2 = demo_individual_packages()
        
        if success2:
            # Demo 3: Performance comparison
            success3 = demo_performance_comparison()
            
            if success3:
                print("\n" + "="*80)
                print("SUCCESS: ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
                print("="*80)
                print("\nThese packages work together to provide:")
                print("• Fast and accurate text preprocessing")
                print("• Improved sentiment analysis performance")
                print("• Consistent text normalization")
                print("• Emotional context preservation")
                print("\nReady to use in your sentiment analysis pipeline!")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("1. Run 'python train.py' to train the sentiment analysis model")
    print("2. Run 'python inference.py' to make predictions")
    print("3. Check the README.md for detailed usage instructions")
    print("="*80)


if __name__ == "__main__":
    main()
