#!/usr/bin/env python3
"""
Interactive demo script for the BERT Philosophical Text Classifier.
This script provides a simple command-line interface for testing the classifier.
"""

import sys
import os
import time

def load_classifier():
    """Load the classifier with proper error handling and progress indication."""
    print("🔄 Initializing BERT classifier...")
    print("⏳ This may take a moment to download BERT model files...")
    
    try:
        from main import PhilosophyClassifier
        return PhilosophyClassifier
    except Exception as e:
        print(f"❌ Failed to import classifier: {e}")
        return None

def print_banner():
    """Print a welcome banner."""
    print("=" * 70)
    print("🎓 BERT PHILOSOPHICAL TEXT CLASSIFIER - INTERACTIVE DEMO")
    print("=" * 70)
    print("This classifier distinguishes between Continental and Analytic philosophy styles.")
    print()
    print("Continental Philosophy:")
    print("  🏛️  Emphasis on lived experience and historical context")
    print("  🌊 Dialectical and phenomenological approaches")
    print("  📖 Focus on interpretation and hermeneutics")
    print()
    print("Analytic Philosophy:")
    print("  🔬 Logical rigor and formal analysis")
    print("  🎯 Conceptual clarity and precision")
    print("  ⚖️  Problem-solving methodology")
    print()

def display_result(result):
    """Display classification results in a user-friendly format."""
    print("\nCLASSIFICATION RESULTS")
    print("-" * 40)
    
    continental_prob = result['Continental'] * 100
    analytic_prob = result['Analytic'] * 100
    predicted_class = 'Continental' if continental_prob > analytic_prob else 'Analytic'
    confidence = max(continental_prob, analytic_prob)
    
    # Create visual bars
    cont_bar = "█" * int(continental_prob / 5) + "░" * (20 - int(continental_prob / 5))
    anal_bar = "█" * int(analytic_prob / 5) + "░" * (20 - int(analytic_prob / 5))
    
    print(f"Continental: {cont_bar} {continental_prob:.1f}%")
    print(f"Analytic:    {anal_bar} {analytic_prob:.1f}%")
    print()
    print(f"Prediction: {predicted_class} (Confidence: {confidence:.1f}%)")
    
def get_sample_texts():
    """Return sample texts for demonstration."""
    return {
        "1": {
            "text": "Being-in-the-world is a fundamental structure of Dasein that reveals the primordial unity of our existence. The phenomenon of anxiety discloses the nothingness that underlies all beings.",
            "description": "Heidegger-style existential analysis",
            "expected": "Continental"
        },
        "2": {
            "text": "If knowledge is justified true belief, then the Gettier problem shows that justification alone is insufficient. We need an additional condition that prevents epistemic luck.",
            "description": "Epistemological analysis",
            "expected": "Analytic"
        },
        "3": {
            "text": "The dialectical movement of history unfolds through the negation of negation, where consciousness encounters its other and returns to itself transformed.",
            "description": "Hegelian dialectical philosophy",
            "expected": "Continental"
        },
        "4": {
            "text": "Modal realism holds that all possible worlds exist as concrete physical realities. This thesis provides elegant solutions to problems about properties and propositions.",
            "description": "Modal logic and metaphysics",
            "expected": "Analytic"
        },
        "5": {
            "text": "Language is the house of Being, and in its home dwells man. We are always already going through words even when we don't speak them aloud.",
            "description": "Heidegger on language and Being",
            "expected": "Continental"
        }
    }

def main():
    """Main interactive demo function."""
    print_banner()
    
    # Try to load the classifier class
    PhilosophyClassifier = load_classifier()
    if PhilosophyClassifier is None:
        print("❌ Failed to load the classifier. Please check your installation.")
        return
    
    # Try to instantiate the classifier
    print("🔧 Creating classifier instance...")
    try:
        classifier = PhilosophyClassifier()
        print("✅ Classifier created successfully")
    except Exception as e:
        print(f"❌ Error creating classifier: {e}")
        print("This might be due to missing dependencies or network issues.")
        return
    
    model_path = "philosophy_bert_model.pth"
    
    if os.path.exists(model_path):
        try:
            classifier.load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please train the model first using main.py --train")
            return
    else:
        print(f"❌ Model file {model_path} not found.")
        print("Please train the model first using: python main.py --train")
        return
    
    sample_texts = get_sample_texts()
    
    while True:
        print("\n" + "=" * 50)
        print("CHOOSE AN OPTION:")
        print("1. Try sample texts")
        print("2. Enter your own text")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n📚 SAMPLE TEXTS:")
            print("-" * 30)
            for key, sample in sample_texts.items():
                print(f"{key}. {sample['description']} (Expected: {sample['expected']})")
            
            sample_choice = input("\nChoose a sample (1-5): ").strip()
            
            if sample_choice in sample_texts:
                sample = sample_texts[sample_choice]
                print(f"\n📝 Text: {sample['text']}")
                print(f"📋 Description: {sample['description']}")
                print(f"🎯 Expected: {sample['expected']}")
                
                try:
                    result = classifier.predict(sample['text'])
                    display_result(result)
                except Exception as e:
                    print(f"❌ Error during prediction: {e}")
            else:
                print("❌ Invalid choice. Please select 1-5.")
        
        elif choice == "2":
            print("\n📝 ENTER YOUR PHILOSOPHICAL TEXT:")
            print("(Type your text and press Enter)")
            print("-" * 40)
            
            user_text = input("Text: ").strip()
            
            if not user_text:
                print("❌ Please enter some text.")
                continue
            
            if len(user_text) < 10:
                print("⚠️  Text seems very short. Results may be less reliable.")
            
            try:
                result = classifier.predict(user_text)
                print(f"\nYour text: {user_text}")
                display_result(result)
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
        
        elif choice == "3":
            print("\n👋 Thank you for using the Philosophical Text Classifier!")
            print("🎓 Happy philosophizing!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        sys.exit(1)
