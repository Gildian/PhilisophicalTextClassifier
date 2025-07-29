#!/usr/bin/env python3
"""
Token-level analysis and visualization for the Philosophical Text Classifier.
This module provides functionality to analyze how individual tokens influence
the classification decision between Continental and Analytic philosophy.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

class TokenAnalyzer:
    """
    Analyzes token-level contributions to philosophical text classification.
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.token_contributions = []
        self.class_names = ['Continental', 'Analytic']
        
    def analyze_token_contributions(self, text: str, return_attention: bool = True) -> Dict:
        """
        Analyze how each token contributes to the classification decision.
        
        Args:
            text: Input text to analyze
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing token analysis results
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
        
        # Enable gradient computation for input embeddings
        input_ids.requires_grad_(True)
        
        # Forward pass with gradient retention
        with torch.enable_grad():
            logits, probabilities = self.model(input_ids, attention_mask)
            
            # Get prediction
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_prob = probabilities[0, predicted_class].item()
            
            # Compute gradients with respect to input embeddings
            loss = F.cross_entropy(logits, torch.tensor([predicted_class]).to(self.device))
            loss.backward()
            
        # Get token information
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_ids = input_ids[0].detach().cpu().numpy()
        
        # Get gradients (proxy for token importance)
        if input_ids.grad is not None:
            gradients = input_ids.grad[0].detach().cpu().numpy()
            token_importance = np.abs(gradients)
        else:
            token_importance = np.zeros(len(tokens))
        
        # Analyze attention patterns if model has attention
        attention_weights = None
        if hasattr(self.model.bert, 'encoder') and return_attention:
            with torch.enable_grad():
                outputs = self.model.bert(input_ids, attention_mask, output_attentions=True)
                if outputs.attentions is not None:
                    # Average attention across all heads and layers
                    attention_weights = torch.stack(outputs.attentions).mean(dim=(0, 1)).detach().cpu().numpy()
        
        return {
            'text': text,
            'tokens': tokens,
            'token_ids': token_ids,
            'token_importance': token_importance,
            'attention_weights': attention_weights,
            'predicted_class': self.class_names[predicted_class],
            'predicted_prob': predicted_prob,
            'probabilities': {
                'Continental': probabilities[0, 0].item(),
                'Analytic': probabilities[0, 1].item()
            }
        }
    
    def visualize_token_contributions(self, analysis_result: Dict, save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive visualization of token contributions.
        
        Args:
            analysis_result: Result from analyze_token_contributions
            save_path: Optional path to save the visualization
            
        Returns:
            Plotly figure object
        """
        tokens = analysis_result['tokens']
        importance = analysis_result['token_importance']
        predicted_class = analysis_result['predicted_class']
        probabilities = analysis_result['probabilities']
        
        # Normalize importance scores for better visualization
        if importance.max() > 0:
            normalized_importance = importance / importance.max()
        else:
            normalized_importance = importance
        
        # Create color scale based on importance
        colors = []
        for imp in normalized_importance:
            if predicted_class == 'Continental':
                # Red shades for Continental
                colors.append(f'rgba(220, 53, 69, {imp * 0.8 + 0.2})')
            else:
                # Blue shades for Analytic
                colors.append(f'rgba(13, 110, 253, {imp * 0.8 + 0.2})')
        
        # Create the figure
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.8, 0.2],
            subplot_titles=(
                f'Token Contributions to {predicted_class} Classification',
                'Class Probabilities'
            ),
            vertical_spacing=0.1
        )
        
        # Token importance visualization
        fig.add_trace(
            go.Bar(
                x=list(range(len(tokens))),
                y=importance,
                text=tokens,
                textposition='outside',
                marker_color=colors,
                hovertemplate='<b>Token:</b> %{text}<br>' +
                            '<b>Importance:</b> %{y:.4f}<br>' +
                            '<extra></extra>',
                name='Token Importance'
            ),
            row=1, col=1
        )
        
        # Class probabilities
        prob_colors = ['#dc3545' if cls == 'Continental' else '#0d6efd' 
                      for cls in probabilities.keys()]
        
        fig.add_trace(
            go.Bar(
                x=list(probabilities.keys()),
                y=list(probabilities.values()),
                marker_color=prob_colors,
                text=[f'{prob:.1%}' for prob in probabilities.values()],
                textposition='auto',
                hovertemplate='<b>Class:</b> %{x}<br>' +
                            '<b>Probability:</b> %{y:.1%}<br>' +
                            '<extra></extra>',
                name='Class Probabilities'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Philosophical Text Classification Analysis<br><sup>Text: "{analysis_result["text"][:100]}..."</sup>',
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update x-axis for token plot
        fig.update_xaxes(
            title='Token Position',
            tickmode='array',
            tickvals=list(range(len(tokens))),
            ticktext=[f'{i}' for i in range(len(tokens))],
            row=1, col=1
        )
        
        fig.update_yaxes(title='Importance Score', row=1, col=1)
        fig.update_yaxes(title='Probability', row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def analyze_batch_tokens(self, texts: List[str], labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze token contributions for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            labels: Optional list of true labels
            
        Returns:
            DataFrame with token analysis results
        """
        results = []
        
        for i, text in enumerate(texts):
            print(f"Analyzing text {i+1}/{len(texts)}...")
            
            try:
                analysis = self.analyze_token_contributions(text)
                
                # Extract important tokens (top 10)
                importance = analysis['token_importance']
                tokens = analysis['tokens']
                
                # Get top contributing tokens
                if len(importance) > 0:
                    top_indices = np.argsort(importance)[-10:][::-1]
                    top_tokens = [tokens[idx] for idx in top_indices]
                    top_scores = [importance[idx] for idx in top_indices]
                else:
                    top_tokens = []
                    top_scores = []
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'predicted_class': analysis['predicted_class'],
                    'continental_prob': analysis['probabilities']['Continental'],
                    'analytic_prob': analysis['probabilities']['Analytic'],
                    'confidence': max(analysis['probabilities'].values()),
                    'top_tokens': ', '.join(top_tokens[:5]),  # Top 5 tokens
                    'avg_token_importance': np.mean(importance) if len(importance) > 0 else 0
                }
                
                if labels:
                    result['true_label'] = labels[i]
                    result['correct'] = labels[i] == analysis['predicted_class']
                
                results.append(result)
                
            except Exception as e:
                print(f"Error analyzing text {i+1}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def create_token_heatmap(self, analysis_result: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap visualization of token importance.
        
        Args:
            analysis_result: Result from analyze_token_contributions
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        tokens = analysis_result['tokens']
        importance = analysis_result['token_importance']
        predicted_class = analysis_result['predicted_class']
        
        # Clean tokens for display (remove special characters)
        display_tokens = []
        for token in tokens:
            if token.startswith('##'):
                display_tokens.append(token[2:])
            elif token in ['[CLS]', '[SEP]', '[PAD]']:
                display_tokens.append(token)
            else:
                display_tokens.append(token)
        
        # Reshape for heatmap (try to make roughly square)
        n_tokens = len(tokens)
        n_cols = min(20, n_tokens)  # Max 20 columns
        n_rows = (n_tokens + n_cols - 1) // n_cols
        
        # Pad arrays if necessary
        padded_tokens = display_tokens + [''] * (n_rows * n_cols - len(display_tokens))
        padded_importance = np.pad(importance, (0, n_rows * n_cols - len(importance)))
        
        # Reshape
        token_matrix = np.array(padded_tokens).reshape(n_rows, n_cols)
        importance_matrix = padded_importance.reshape(n_rows, n_cols)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(max(12, n_cols * 0.8), max(6, n_rows * 0.6)))
        
        # Choose colormap based on predicted class
        cmap = 'Reds' if predicted_class == 'Continental' else 'Blues'
        
        # Create heatmap
        sns.heatmap(
            importance_matrix,
            annot=token_matrix,
            fmt='',
            cmap=cmap,
            cbar_kws={'label': 'Token Importance'},
            ax=ax,
            annot_kws={'size': 8, 'weight': 'bold'}
        )
        
        ax.set_title(f'Token Importance Heatmap - {predicted_class} Classification\n'
                    f'Text: "{analysis_result["text"][:80]}..."', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Position (Column)')
        ax.set_ylabel('Token Position (Row)')
        
        # Remove tick labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        return fig
    
    def save_analysis_report(self, analysis_results: List[Dict], output_path: str):
        """
        Save a comprehensive analysis report to HTML.
        
        Args:
            analysis_results: List of analysis results
            output_path: Path to save the HTML report
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Philosophical Text Classification - Token Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .analysis {{ margin: 30px 0; border: 1px solid #ddd; padding: 20px; border-radius: 8px; }}
                .token {{ display: inline-block; margin: 2px; padding: 3px 6px; border-radius: 3px; }}
                .continental {{ background-color: #ffebee; border: 1px solid #f44336; }}
                .analytic {{ background-color: #e3f2fd; border: 1px solid #2196f3; }}
                .high-importance {{ font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎓 Philosophical Text Classification</h1>
                <h2>Token-Level Analysis Report</h2>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        for i, result in enumerate(analysis_results):
            tokens = result['tokens']
            importance = result['token_importance']
            predicted_class = result['predicted_class']
            probabilities = result['probabilities']
            
            class_style = 'continental' if predicted_class == 'Continental' else 'analytic'
            
            html_content += f"""
            <div class="analysis">
                <h3>Analysis {i+1}</h3>
                <p><strong>Text:</strong> {result['text']}</p>
                <p><strong>Predicted Class:</strong> <span class="{class_style}">{predicted_class}</span></p>
                <p><strong>Confidence:</strong> {max(probabilities.values()):.1%}</p>
                
                <table>
                    <tr>
                        <th>Continental Probability</th>
                        <th>Analytic Probability</th>
                    </tr>
                    <tr>
                        <td>{probabilities['Continental']:.1%}</td>
                        <td>{probabilities['Analytic']:.1%}</td>
                    </tr>
                </table>
                
                <h4>Token Contributions:</h4>
                <div>
            """
            
            # Add tokens with importance-based styling
            for token, imp in zip(tokens, importance):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                    
                opacity = min(1.0, imp / max(importance) if max(importance) > 0 else 0.1)
                high_importance = "high-importance" if imp > np.mean(importance) else ""
                
                html_content += f'<span class="token {class_style} {high_importance}" style="opacity: {opacity}">{token}</span>'
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Analysis report saved to {output_path}")

def create_training_token_analyzer(classifier, sample_texts: List[str], sample_labels: List[str], 
                                 output_dir: str = "token_analysis_results"):
    """
    Create token analysis during training process.
    
    Args:
        classifier: Trained PhilosophyClassifier instance
        sample_texts: List of sample texts to analyze
        sample_labels: List of corresponding labels
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = TokenAnalyzer(classifier.model, classifier.tokenizer, classifier.device)
    
    print("🔍 Starting token-level analysis...")
    
    # Analyze individual texts
    analysis_results = []
    for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
        print(f"Analyzing sample {i+1}/{len(sample_texts)}: {label}")
        
        result = analyzer.analyze_token_contributions(text)
        result['true_label'] = label
        result['correct'] = result['predicted_class'] == label
        analysis_results.append(result)
        
        # Create individual visualizations
        fig = analyzer.visualize_token_contributions(result, 
                                                   f"{output_dir}/token_analysis_{i+1}.html")
        
        # Create heatmap
        heatmap_fig = analyzer.create_token_heatmap(result, 
                                                  f"{output_dir}/token_heatmap_{i+1}.png")
        plt.close(heatmap_fig)
    
    # Create batch analysis
    batch_df = analyzer.analyze_batch_tokens(sample_texts, sample_labels)
    batch_df.to_csv(f"{output_dir}/batch_token_analysis.csv", index=False)
    
    # Create comprehensive report
    analyzer.save_analysis_report(analysis_results, f"{output_dir}/token_analysis_report.html")
    
    print(f"✅ Token analysis complete! Results saved to {output_dir}/")
    print(f"📊 Open {output_dir}/token_analysis_report.html to view the comprehensive report")
    
    return analysis_results, batch_df

if __name__ == "__main__":
    # Example usage
    print("Token Analysis Module - Philosophical Text Classifier")
    print("Import this module and use create_training_token_analyzer() after training your model.")
