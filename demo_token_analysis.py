#!/usr/bin/env python3
"""
Demo script showing how to use the token analysis feature with the Philosophical Text Classifier.
"""

import os
import sys
import matplotlib.pyplot as plt
from main import PhilosophyClassifier

# Try to import the analysis modules
try:
    from simple_token_analysis import create_simple_token_analysis, SimpleTokenAnalyzer
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    print("❌ Token analysis modules not available")

def demo_token_analysis():
    """
    Demonstrate token analysis on sample philosophical texts.
    """
    print("🎓 PHILOSOPHICAL TEXT CLASSIFIER - TOKEN ANALYSIS DEMO")
    print("=" * 60)
    
    # Check if model exists
    model_path = "philosophy_bert_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found.")
        print("Please train the model first using: python main.py --train")
        return
    
    if not ANALYSIS_AVAILABLE:
        print("❌ Token analysis not available. Please check your installation.")
        return
    
    # Load the trained model
    print("🔧 Loading trained model...")
    classifier = PhilosophyClassifier()
    try:
        classifier.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Sample texts for analysis
    sample_texts = [
        "Being-in-the-world is a fundamental structure of Dasein that reveals the primordial unity of our existence. The phenomenon of anxiety discloses the nothingness that underlies all beings.",
        "If knowledge is justified true belief, then the Gettier problem shows that justification alone is insufficient. We need an additional condition that prevents epistemic luck.",
        "The dialectical movement of history unfolds through the negation of negation, where consciousness encounters its other and returns to itself transformed.",
        "Modal realism holds that all possible worlds exist as concrete physical realities. This thesis provides elegant solutions to problems about properties and propositions.",
        "Language is the house of Being, and in its home dwells man. We are always already going through words even when we don't speak them aloud."
    ]
    
    sample_labels = [
        "Continental",
        "Analytic", 
        "Continental",
        "Analytic",
        "Continental"
    ]
    
    print(f"\n🔍 Analyzing {len(sample_texts)} sample texts...")
    print("This may take a moment...")
    
    try:
        # Perform token analysis
        analysis_results, results_df = create_simple_token_analysis(
            classifier,
            sample_texts,
            sample_labels,
            "demo_token_analysis"
        )
        
        print("\n📊 ANALYSIS RESULTS:")
        print("-" * 40)
        print(results_df[['true_label', 'predicted_class', 'correct', 'confidence']].to_string(index=False))
        
        print(f"\n🎯 SUMMARY STATISTICS:")
        print(f"Overall Accuracy: {results_df['correct'].mean():.1%}")
        print(f"Average Confidence: {results_df['confidence'].mean():.1%}")
        print(f"Continental Samples: {len(results_df[results_df['true_label'] == 'Continental'])}")
        print(f"Analytic Samples: {len(results_df[results_df['true_label'] == 'Analytic'])}")
        
        # Show individual token analysis for first sample
        print(f"\n🔬 DETAILED TOKEN ANALYSIS - Sample 1:")
        print("-" * 50)
        first_result = analysis_results[0]
        print(f"Text: {first_result['text'][:100]}...")
        print(f"True Label: {first_result['true_label']}")
        print(f"Predicted: {first_result['predicted_class']}")
        print(f"Continental Prob: {first_result['probabilities']['Continental']:.1%}")
        print(f"Analytic Prob: {first_result['probabilities']['Analytic']:.1%}")
        
        # Show top contributing tokens
        tokens = first_result['tokens']
        importance = first_result['token_importance']
        
        if len(importance) > 0:
            top_indices = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:10]
            print(f"\nTop Contributing Tokens:")
            for i, idx in enumerate(top_indices, 1):
                token = tokens[idx]
                if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                    print(f"  {i:2d}. {token:<15} (importance: {importance[idx]:.4f})")
        
        print(f"\n📁 Detailed visualizations saved to: demo_token_analysis/")
        print(f"🖼️  Check demo_token_analysis/token_summary.png for overview")
        print(f"🖼️  Individual analyses: demo_token_analysis/token_analysis_*.png")
        
        # Ask if user wants to analyze custom text
        print(f"\n" + "="*60)
        custom_analysis = input("Would you like to analyze your own text? (y/n): ").strip().lower()
        
        if custom_analysis == 'y':
            analyze_custom_text(classifier)
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def analyze_custom_text(classifier):
    """
    Allow user to input custom text for token analysis.
    """
    print(f"\n📝 CUSTOM TEXT ANALYSIS")
    print("-" * 30)
    
    while True:
        print(f"\nEnter philosophical text to analyze (or 'quit' to exit):")
        user_text = input("> ").strip()
        
        if user_text.lower() == 'quit':
            break
            
        if not user_text:
            print("❌ Please enter some text.")
            continue
            
        if len(user_text) < 10:
            print("⚠️  Text seems very short. Results may be less reliable.")
        
        try:
            # Create analyzer for single text
            analyzer = SimpleTokenAnalyzer(classifier.model, classifier.tokenizer, classifier.device)
            result = analyzer.analyze_token_contributions(user_text)
            
            print(f"\n🎯 ANALYSIS RESULTS:")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Continental Prob: {result['probabilities']['Continental']:.1%}")
            print(f"Analytic Prob: {result['probabilities']['Analytic']:.1%}")
            print(f"Confidence: {max(result['probabilities'].values()):.1%}")
            
            # Show top tokens
            tokens = result['tokens']
            importance = result['token_importance']
            
            if len(importance) > 0:
                top_indices = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:8]
                print(f"\nTop Contributing Tokens:")
                for i, idx in enumerate(top_indices, 1):
                    token = tokens[idx]
                    if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                        print(f"  {i}. {token:<15} (importance: {importance[idx]:.4f})")
            
            # Create visualization
            fig = analyzer.create_token_visualization(result)
            plt.show()
            
        except Exception as e:
            print(f"❌ Error analyzing text: {e}")

def main():
    """Main function."""
    try:
        demo_token_analysis()
    except KeyboardInterrupt:
        print(f"\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
