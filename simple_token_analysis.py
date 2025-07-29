#!/usr/bin/env python3
"""
Simplified token analysis module using only matplotlib (no plotly dependency).
This provides token-level visualization during training for the Philosophical Text Classifier.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
import io
import base64
from datetime import datetime

class SimpleTokenAnalyzer:
    """
    Simplified token analyzer using only matplotlib for visualization.
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = ['Continental', 'Analytic']
        
    def analyze_token_contributions(self, text: str) -> Dict:
        """
        Analyze how each token contributes to the classification decision.
        """
        self.model.eval()
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits, probabilities = self.model(input_ids, attention_mask)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_prob = probabilities[0, predicted_class].item()
        
        # Compute token importance using gradient-based method
        input_ids.requires_grad_(True)
        
        with torch.enable_grad():
            logits, _ = self.model(input_ids, attention_mask)
            target_class_logit = logits[0, predicted_class]
            target_class_logit.backward()
        
        # Get token information
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_ids = input_ids[0].detach().cpu().numpy()
        
        # Get gradients as importance scores
        if input_ids.grad is not None:
            gradients = input_ids.grad[0].detach().cpu().numpy()
            token_importance = np.abs(gradients)
        else:
            token_importance = np.zeros(len(tokens))
        
        return {
            'text': text,
            'tokens': tokens,
            'token_ids': token_ids,
            'token_importance': token_importance,
            'predicted_class': self.class_names[predicted_class],
            'predicted_prob': predicted_prob,
            'probabilities': {
                'Continental': probabilities[0, 0].item(),
                'Analytic': probabilities[0, 1].item()
            }
        }
    
    def create_token_visualization(self, analysis_result: Dict, save_path: Optional[str] = None, figsize=(15, 10)):
        """
        Create comprehensive token visualization using matplotlib.
        """
        tokens = analysis_result['tokens']
        importance = analysis_result['token_importance']
        predicted_class = analysis_result['predicted_class']
        probabilities = analysis_result['probabilities']
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main token importance plot
        ax1 = fig.add_subplot(gs[0, :])
        
        # Clean tokens for display
        display_tokens = []
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                display_tokens.append(f"{i}:{token[2:]}")
            elif token in ['[CLS]', '[SEP]', '[PAD]']:
                display_tokens.append(f"{i}:{token}")
            else:
                display_tokens.append(f"{i}:{token}")
        
        # Color coding based on predicted class
        colors = ['#dc3545' if predicted_class == 'Continental' else '#0d6efd' for _ in tokens]
        
        # Create bar plot
        bars = ax1.bar(range(len(tokens)), importance, color=colors, alpha=0.7)
        ax1.set_title(f'Token Importance for {predicted_class} Classification\n'
                     f'Text: "{analysis_result["text"][:80]}..."', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Importance Score')
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars for top tokens
        if len(importance) > 0:
            top_indices = np.argsort(importance)[-5:]  # Top 5 tokens
            for idx in top_indices:
                ax1.text(idx, importance[idx] + max(importance) * 0.01, 
                        f'{importance[idx]:.3f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Class probabilities
        ax2 = fig.add_subplot(gs[1, 0])
        prob_colors = ['#dc3545', '#0d6efd']
        bars2 = ax2.bar(probabilities.keys(), probabilities.values(), color=prob_colors, alpha=0.7)
        ax2.set_title('Class Probabilities', fontweight='bold')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Add probability labels
        for bar, prob in zip(bars2, probabilities.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Top contributing tokens
        ax3 = fig.add_subplot(gs[1, 1])
        if len(importance) > 0:
            top_indices = np.argsort(importance)[-8:][::-1]  # Top 8 tokens
            top_tokens = [tokens[i] if not tokens[i].startswith('##') else tokens[i][2:] 
                         for i in top_indices]
            top_scores = [importance[i] for i in top_indices]
            
            bars3 = ax3.barh(range(len(top_tokens)), top_scores, 
                           color=['#dc3545' if predicted_class == 'Continental' else '#0d6efd'] * len(top_tokens),
                           alpha=0.7)
            ax3.set_yticks(range(len(top_tokens)))
            ax3.set_yticklabels(top_tokens, fontsize=10)
            ax3.set_xlabel('Importance Score')
            ax3.set_title('Top Contributing Tokens', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add score labels
            for i, score in enumerate(top_scores):
                ax3.text(score + max(top_scores) * 0.01, i, f'{score:.3f}',
                        va='center', fontsize=8)
        
        # Token heatmap
        ax4 = fig.add_subplot(gs[2, :])
        if len(tokens) > 0:
            # Arrange tokens in a grid for heatmap
            n_cols = min(25, len(tokens))
            n_rows = (len(tokens) + n_cols - 1) // n_cols
            
            # Pad arrays
            padded_importance = np.pad(importance, (0, n_rows * n_cols - len(importance)))
            importance_matrix = padded_importance.reshape(n_rows, n_cols)
            
            # Create heatmap
            cmap = 'Reds' if predicted_class == 'Continental' else 'Blues'
            im = ax4.imshow(importance_matrix, cmap=cmap, aspect='auto')
            
            ax4.set_title('Token Importance Heatmap', fontweight='bold')
            ax4.set_xlabel('Token Position (columns)')
            ax4.set_ylabel('Token Position (rows)')
            
            # Add colorbar
            plt.colorbar(im, ax=ax4, label='Importance Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Token visualization saved to {save_path}")
        
        return fig
    
    def create_token_summary_plot(self, analysis_results: List[Dict], save_path: Optional[str] = None):
        """
        Create a summary plot showing token analysis across multiple texts.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect data
        continental_correct = []
        analytic_correct = []
        all_confidences = []
        continental_confidences = []
        analytic_confidences = []
        
        # Collect top tokens for each class
        continental_tokens = {}
        analytic_tokens = {}
        
        for result in analysis_results:
            true_label = result.get('true_label', 'Unknown')
            predicted_class = result['predicted_class']
            confidence = max(result['probabilities'].values())
            
            all_confidences.append(confidence)
            
            if predicted_class == 'Continental':
                continental_confidences.append(confidence)
                if true_label == 'Continental':
                    continental_correct.append(1)
                else:
                    continental_correct.append(0)
            else:
                analytic_confidences.append(confidence)
                if true_label == 'Analytic':
                    analytic_correct.append(1)
                else:
                    analytic_correct.append(0)
            
            # Collect important tokens
            tokens = result['tokens']
            importance = result['token_importance']
            
            if len(importance) > 0:
                top_indices = np.argsort(importance)[-5:]  # Top 5 tokens
                for idx in top_indices:
                    token = tokens[idx]
                    if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                        if predicted_class == 'Continental':
                            continental_tokens[token] = continental_tokens.get(token, 0) + importance[idx]
                        else:
                            analytic_tokens[token] = analytic_tokens.get(token, 0) + importance[idx]
        
        # Plot 1: Accuracy by class
        classes = []
        accuracies = []
        if continental_correct:
            classes.append('Continental')
            accuracies.append(np.mean(continental_correct))
        if analytic_correct:
            classes.append('Analytic')
            accuracies.append(np.mean(analytic_correct))
        
        if classes:
            bars1 = ax1.bar(classes, accuracies, color=['#dc3545', '#0d6efd'], alpha=0.7)
            ax1.set_title('Classification Accuracy by Class', fontweight='bold')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            for bar, acc in zip(bars1, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Confidence distribution
        if all_confidences:
            ax2.hist(all_confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_confidences):.1%}')
            ax2.set_title('Confidence Score Distribution', fontweight='bold')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        
        # Plot 3: Top Continental tokens
        if continental_tokens:
            top_continental = sorted(continental_tokens.items(), key=lambda x: x[1], reverse=True)[:10]
            tokens_c, scores_c = zip(*top_continental)
            ax3.barh(range(len(tokens_c)), scores_c, color='#dc3545', alpha=0.7)
            ax3.set_yticks(range(len(tokens_c)))
            ax3.set_yticklabels(tokens_c)
            ax3.set_title('Top Continental Philosophy Tokens', fontweight='bold')
            ax3.set_xlabel('Cumulative Importance Score')
        
        # Plot 4: Top Analytic tokens
        if analytic_tokens:
            top_analytic = sorted(analytic_tokens.items(), key=lambda x: x[1], reverse=True)[:10]
            tokens_a, scores_a = zip(*top_analytic)
            ax4.barh(range(len(tokens_a)), scores_a, color='#0d6efd', alpha=0.7)
            ax4.set_yticks(range(len(tokens_a)))
            ax4.set_yticklabels(tokens_a)
            ax4.set_title('Top Analytic Philosophy Tokens', fontweight='bold')
            ax4.set_xlabel('Cumulative Importance Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Token summary plot saved to {save_path}")
        
        return fig

def create_simple_token_analysis(classifier, sample_texts: List[str], sample_labels: List[str], 
                                output_dir: str = "simple_token_analysis"):
    """
    Create simplified token analysis using only matplotlib.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Starting simplified token analysis...")
    
    # Initialize analyzer
    analyzer = SimpleTokenAnalyzer(classifier.model, classifier.tokenizer, classifier.device)
    
    # Analyze texts
    analysis_results = []
    for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
        print(f"Analyzing sample {i+1}/{len(sample_texts)}: {label}")
        
        result = analyzer.analyze_token_contributions(text)
        result['true_label'] = label
        result['correct'] = result['predicted_class'] == label
        analysis_results.append(result)
        
        # Create individual visualization
        fig = analyzer.create_token_visualization(result, 
                                                f"{output_dir}/token_analysis_{i+1}.png")
        plt.close(fig)
    
    # Create summary plot
    summary_fig = analyzer.create_token_summary_plot(analysis_results,
                                                   f"{output_dir}/token_summary.png")
    plt.close(summary_fig)
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
            'true_label': r['true_label'],
            'predicted_class': r['predicted_class'],
            'correct': r['correct'],
            'continental_prob': r['probabilities']['Continental'],
            'analytic_prob': r['probabilities']['Analytic'],
            'confidence': max(r['probabilities'].values()),
            'avg_token_importance': np.mean(r['token_importance']) if len(r['token_importance']) > 0 else 0
        }
        for r in analysis_results
    ])
    
    results_df.to_csv(f"{output_dir}/analysis_results.csv", index=False)
    
    print(f"✅ Simple token analysis complete! Results saved to {output_dir}/")
    print(f"📊 Check {output_dir}/token_summary.png for overview")
    
    return analysis_results, results_df

if __name__ == "__main__":
    print("Simple Token Analysis Module - Philosophical Text Classifier")
    print("This module provides matplotlib-based token visualization.")
