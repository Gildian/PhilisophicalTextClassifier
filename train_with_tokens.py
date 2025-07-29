#!/usr/bin/env python3
"""
Enhanced training script with token-level analysis for the Philosophical Text Classifier.
This script extends the main training functionality to include token visualization during training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import PhilosophyClassifier
try:
    from token_analysis import create_training_token_analyzer, TokenAnalyzer
    PLOTLY_AVAILABLE = True
except ImportError:
    from simple_token_analysis import create_simple_token_analysis, SimpleTokenAnalyzer
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available, using simplified matplotlib-based token analysis")
import pandas as pd
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def train_with_token_analysis(data_path: str, model_save_path: str = "philosophy_bert_model.pth", 
                             epochs: int = 3, enable_token_analysis: bool = True,
                             analysis_samples: int = 10):
    """
    Train the model with integrated token-level analysis.
    
    Args:
        data_path: Path to training data (CSV or JSON)
        model_save_path: Path to save the trained model
        epochs: Number of training epochs
        enable_token_analysis: Whether to perform token analysis
        analysis_samples: Number of samples to analyze in detail
    """
    print("🚀 Starting Enhanced Training with Token Analysis")
    print("=" * 60)
    
    # Initialize classifier
    classifier = PhilosophyClassifier()
    
    # Load data
    if data_path.endswith('.csv'):
        df = classifier.load_data_from_csv(data_path)
    elif data_path.endswith('.json'):
        df = classifier.load_data_from_json(data_path)
    else:
        raise ValueError("Data file must be CSV or JSON format")
    
    # Prepare data
    train_loader, val_loader = classifier.prepare_data(df)
    
    print(f"\n📊 Training Data Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Continental: {len(df[df['label'] == 'Continental'])}")
    print(f"Analytic: {len(df[df['label'] == 'Analytic'])}")
    print(f"Training epochs: {epochs}")
    
    # Train the model
    print("\n🔥 Starting Model Training...")
    train_losses, val_losses, val_accuracies = classifier.train_model(
        train_loader, val_loader, epochs=epochs
    )
    
    # Save the model
    classifier.save_model(model_save_path)
    print(f"✅ Model saved to {model_save_path}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_accuracies)
    
    if enable_token_analysis and len(df) > 0:
        print("\n🔍 Starting Token-Level Analysis...")
        print("=" * 50)
        
        # Select samples for analysis
        analysis_df = select_analysis_samples(df, analysis_samples)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"token_analysis_{timestamp}"
        
        # Perform token analysis
        try:
            if PLOTLY_AVAILABLE:
                analysis_results, batch_results = create_training_token_analyzer(
                    classifier, 
                    analysis_df['text'].tolist(), 
                    analysis_df['label'].tolist(),
                    output_dir
                )
                
                # Generate analysis summary
                generate_analysis_summary(analysis_results, batch_results, output_dir)
                
                print(f"\n🎯 Token Analysis Summary:")
                print(f"Samples analyzed: {len(analysis_results)}")
                print(f"Accuracy on analyzed samples: {batch_results['correct'].mean():.1%}")
                print(f"Average confidence: {batch_results['confidence'].mean():.1%}")
                
                # Show most important tokens
                show_top_tokens(analysis_results)
            else:
                # Use simplified analysis
                analysis_results, batch_results = create_simple_token_analysis(
                    classifier,
                    analysis_df['text'].tolist(), 
                    analysis_df['label'].tolist(),
                    output_dir
                )
                
                print(f"\n🎯 Simplified Token Analysis Summary:")
                print(f"Samples analyzed: {len(analysis_results)}")
                print(f"Accuracy on analyzed samples: {batch_results['correct'].mean():.1%}")
                print(f"Average confidence: {batch_results['confidence'].mean():.1%}")
            
        except Exception as e:
            print(f"❌ Token analysis failed: {e}")
            print("Continuing without token analysis...")
    
    print("\n🎉 Training Complete!")
    return classifier

def select_analysis_samples(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """
    Select representative samples for token analysis.
    """
    # Try to get balanced samples
    continental_samples = df[df['label'] == 'Continental'].sample(
        min(n_samples // 2, len(df[df['label'] == 'Continental'])), 
        random_state=42
    )
    analytic_samples = df[df['label'] == 'Analytic'].sample(
        min(n_samples // 2, len(df[df['label'] == 'Analytic'])), 
        random_state=42
    )
    
    analysis_df = pd.concat([continental_samples, analytic_samples]).reset_index(drop=True)
    
    # If we need more samples, randomly select from remaining
    if len(analysis_df) < n_samples:
        remaining = df[~df.index.isin(analysis_df.index)]
        additional = remaining.sample(
            min(n_samples - len(analysis_df), len(remaining)),
            random_state=42
        )
        analysis_df = pd.concat([analysis_df, additional]).reset_index(drop=True)
    
    return analysis_df[:n_samples]

def plot_training_curves(train_losses, val_losses, val_accuracies):
    """
    Plot training curves with enhanced visualization.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Training and validation loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2, marker='^')
    ax2.set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Loss comparison
    ax3.bar(['Training', 'Validation'], [train_losses[-1], val_losses[-1]], 
            color=['blue', 'red'], alpha=0.7)
    ax3.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Loss')
    
    # Performance metrics
    final_accuracy = val_accuracies[-1]
    ax4.pie([final_accuracy, 1-final_accuracy], 
            labels=['Correct', 'Incorrect'], 
            colors=['green', 'red'], 
            autopct='%1.1f%%',
            startangle=90)
    ax4.set_title(f'Final Validation Accuracy\n{final_accuracy:.1%}', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"📈 Training curves saved to training_curves_{timestamp}.png")
    
    plt.show()

def generate_analysis_summary(analysis_results, batch_results, output_dir):
    """
    Generate a comprehensive analysis summary.
    """
    summary_path = f"{output_dir}/analysis_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("PHILOSOPHICAL TEXT CLASSIFIER - TOKEN ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples Analyzed: {len(analysis_results)}\n\n")
        
        # Accuracy metrics
        f.write("ACCURACY METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Overall Accuracy: {batch_results['correct'].mean():.1%}\n")
        f.write(f"Continental Accuracy: {batch_results[batch_results['true_label'] == 'Continental']['correct'].mean():.1%}\n")
        f.write(f"Analytic Accuracy: {batch_results[batch_results['true_label'] == 'Analytic']['correct'].mean():.1%}\n\n")
        
        # Confidence metrics
        f.write("CONFIDENCE METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average Confidence: {batch_results['confidence'].mean():.1%}\n")
        f.write(f"Median Confidence: {batch_results['confidence'].median():.1%}\n")
        f.write(f"Min Confidence: {batch_results['confidence'].min():.1%}\n")
        f.write(f"Max Confidence: {batch_results['confidence'].max():.1%}\n\n")
        
        # Detailed results
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 20 + "\n")
        for i, result in enumerate(analysis_results):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Text: {result['text'][:100]}...\n")
            f.write(f"  True Label: {result['true_label']}\n")
            f.write(f"  Predicted: {result['predicted_class']}\n")
            f.write(f"  Correct: {result['correct']}\n")
            f.write(f"  Continental Prob: {result['probabilities']['Continental']:.1%}\n")
            f.write(f"  Analytic Prob: {result['probabilities']['Analytic']:.1%}\n\n")
    
    print(f"📋 Analysis summary saved to {summary_path}")

def show_top_tokens(analysis_results):
    """
    Display the most important tokens across all analyses.
    """
    print("\n🏆 TOP CONTRIBUTING TOKENS:")
    print("-" * 40)
    
    # Collect all tokens and their importance scores
    all_token_importance = {}
    
    for result in analysis_results:
        tokens = result['tokens']
        importance = result['token_importance']
        predicted_class = result['predicted_class']
        
        for token, imp in zip(tokens, importance):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]'] or token.startswith('##'):
                continue
            
            key = f"{token}_{predicted_class}"
            if key not in all_token_importance:
                all_token_importance[key] = []
            all_token_importance[key].append(imp)
    
    # Calculate average importance for each token-class pair
    token_avg_importance = {}
    for key, importance_list in all_token_importance.items():
        token_avg_importance[key] = np.mean(importance_list)
    
    # Sort and display top tokens
    sorted_tokens = sorted(token_avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("Continental Philosophy Tokens:")
    continental_tokens = [(k, v) for k, v in sorted_tokens if k.endswith('_Continental')][:10]
    for i, (token_class, importance) in enumerate(continental_tokens, 1):
        token = token_class.replace('_Continental', '')
        print(f"  {i:2d}. {token:<15} (importance: {importance:.4f})")
    
    print("\nAnalytic Philosophy Tokens:")
    analytic_tokens = [(k, v) for k, v in sorted_tokens if k.endswith('_Analytic')][:10]
    for i, (token_class, importance) in enumerate(analytic_tokens, 1):
        token = token_class.replace('_Analytic', '')
        print(f"  {i:2d}. {token:<15} (importance: {importance:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Train Philosophical Text Classifier with Token Analysis')
    parser.add_argument('--data', required=True, help='Path to training data (CSV or JSON)')
    parser.add_argument('--model', default='philosophy_bert_model.pth', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--no-token-analysis', action='store_true', help='Disable token analysis')
    parser.add_argument('--analysis-samples', type=int, default=10, help='Number of samples for token analysis')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)
    
    # Train with token analysis
    classifier = train_with_token_analysis(
        data_path=args.data,
        model_save_path=args.model,
        epochs=args.epochs,
        enable_token_analysis=not args.no_token_analysis,
        analysis_samples=args.analysis_samples
    )
    
    print(f"\n🎯 Model training complete!")
    print(f"Run 'python demo.py' to test your trained model.")

if __name__ == "__main__":
    main()
