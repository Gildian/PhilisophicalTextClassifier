#!/usr/bin/env python3
"""
Interactive Jupyter notebook interface for the BERT Philosophical Text Classifier.
This module provides utility functions for notebook-based exploration and analysis.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import json

# Import our main classifier
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import PhilosophyClassifier

class InteractivePhilosophyClassifier:
    """
    Interactive wrapper for the philosophy classifier with Jupyter notebook support.
    """
    
    def __init__(self):
        self.classifier = PhilosophyClassifier()
        self.model_loaded = False
        
    def create_text_input_widget(self):
        """Create an interactive text input widget for classification."""
        
        # Text input
        text_input = widgets.Textarea(
            value='',
            placeholder='Enter philosophical text to classify...',
            description='Text:',
            disabled=False,
            layout=widgets.Layout(width='80%', height='150px')
        )
        
        # Classify button
        classify_button = widgets.Button(
            description='Classify Text',
            disabled=False,
            button_style='primary',
            tooltip='Click to classify the text',
            icon='check'
        )
        
        # Output area
        output = widgets.Output()
        
        def on_classify_click(b):
            with output:
                clear_output()
                if not self.model_loaded:
                    print("❌ Model not loaded. Please load a trained model first.")
                    return
                
                text = text_input.value.strip()
                if not text:
                    print("❌ Please enter some text to classify.")
                    return
                
                try:
                    result = self.classifier.predict(text)
                    self.display_classification_result(result, text)
                except Exception as e:
                    print(f"❌ Error during classification: {str(e)}")
        
        classify_button.on_click(on_classify_click)
        
        # Display widgets
        display(text_input)
        display(classify_button)
        display(output)
        
        return text_input, classify_button, output
    
    def display_classification_result(self, result: Dict[str, float], text: str):
        """Display classification results in a formatted way."""
        
        # Create a visual representation
        continental_prob = result['Continental'] * 100
        analytic_prob = result['Analytic'] * 100
        
        # Determine prediction
        predicted_class = 'Continental' if continental_prob > analytic_prob else 'Analytic'
        confidence = max(continental_prob, analytic_prob)
        
        # Create HTML output
        html_output = f"""
        <div style="border: 2px solid #ddd; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f9f9f9;">
            <h3 style="color: #333; margin-top: 0;">📊 Classification Results</h3>
            
            <div style="margin: 15px 0;">
                <strong>Input Text:</strong> 
                <em>"{text[:100]}{'...' if len(text) > 100 else ''}"</em>
            </div>
            
            <div style="margin: 15px 0;">
                <div style="display: flex; align-items: center; margin: 10px 0;">
                    <span style="width: 120px; font-weight: bold;">🏛️ Continental:</span>
                    <div style="background-color: #e0e0e0; width: 200px; height: 20px; border-radius: 10px; margin: 0 10px;">
                        <div style="background-color: #ff6b6b; width: {continental_prob}%; height: 100%; border-radius: 10px;"></div>
                    </div>
                    <span style="font-weight: bold; color: #ff6b6b;">{continental_prob:.1f}%</span>
                </div>
                
                <div style="display: flex; align-items: center; margin: 10px 0;">
                    <span style="width: 120px; font-weight: bold;">🔬 Analytic:</span>
                    <div style="background-color: #e0e0e0; width: 200px; height: 20px; border-radius: 10px; margin: 0 10px;">
                        <div style="background-color: #4ecdc4; width: {analytic_prob}%; height: 100%; border-radius: 10px;"></div>
                    </div>
                    <span style="font-weight: bold; color: #4ecdc4;">{analytic_prob:.1f}%</span>
                </div>
            </div>
            
            <div style="margin: 15px 0; padding: 10px; background-color: {'#ffe6e6' if predicted_class == 'Continental' else '#e6f7f7'}; border-radius: 5px;">
                <strong>🎯 Prediction:</strong> 
                <span style="color: {'#cc0000' if predicted_class == 'Continental' else '#006666'}; font-size: 1.2em;">
                    {predicted_class}
                </span>
                <span style="color: #666;">(Confidence: {confidence:.1f}%)</span>
            </div>
        </div>
        """
        
        display(HTML(html_output))
    
    def load_model_interactive(self, model_path: str = 'philosophy_bert_model.pth'):
        """Load a trained model with user feedback."""
        try:
            if os.path.exists(model_path):
                self.classifier.load_model(model_path)
                self.model_loaded = True
                print(f"✅ Model loaded successfully from {model_path}")
                return True
            else:
                print(f"❌ Model file not found: {model_path}")
                print("Please train a model first using the training notebook or main.py")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def create_training_widget(self):
        """Create an interactive training interface."""
        
        # Training parameters
        epochs_widget = widgets.IntSlider(
            value=3,
            min=1,
            max=10,
            step=1,
            description='Epochs:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        learning_rate_widget = widgets.SelectionSlider(
            options=[1e-5, 2e-5, 3e-5, 5e-5],
            value=2e-5,
            description='Learning Rate:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )
        
        # Train button
        train_button = widgets.Button(
            description='Start Training',
            disabled=False,
            button_style='success',
            tooltip='Click to start training',
            icon='play'
        )
        
        # Output area
        output = widgets.Output()
        
        def on_train_click(b):
            with output:
                clear_output()
                print("🚀 Starting training process...")
                
                try:
                    # Create sample data
                    df = self.classifier.create_sample_data()
                    print(f"📊 Created dataset with {len(df)} samples")
                    
                    # Prepare data
                    train_loader, val_loader = self.classifier.prepare_data(df)
                    print("📋 Data prepared successfully")
                    
                    # Train model
                    print("🎯 Training model...")
                    train_losses, val_losses, val_accuracies = self.classifier.train_model(
                        train_loader, val_loader, 
                        epochs=epochs_widget.value,
                        learning_rate=learning_rate_widget.value
                    )
                    
                    # Save model
                    model_path = 'philosophy_bert_model.pth'
                    self.classifier.save_model(model_path)
                    self.model_loaded = True
                    
                    print("✅ Training completed successfully!")
                    
                    # Plot training curves
                    self.plot_training_curves(train_losses, val_losses, val_accuracies)
                    
                except Exception as e:
                    print(f"❌ Error during training: {str(e)}")
        
        train_button.on_click(on_train_click)
        
        # Display widgets
        print("🔧 Training Configuration:")
        display(epochs_widget)
        display(learning_rate_widget)
        display(train_button)
        display(output)
        
        return epochs_widget, learning_rate_widget, train_button, output
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        """Plot training curves in the notebook."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', marker='o')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', marker='s')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', marker='^')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_sample_texts(self):
        """Analyze and display predictions for sample texts."""
        
        if not self.model_loaded:
            print("❌ Model not loaded. Please load a trained model first.")
            return
        
        # Sample texts for analysis
        sample_texts = {
            "Continental 1": "Being-in-the-world is a fundamental structure of Dasein that reveals the primordial unity of our existence.",
            "Continental 2": "The dialectical movement of history unfolds through the negation of negation, where consciousness encounters its other.",
            "Analytic 1": "If knowledge is justified true belief, then the Gettier problem shows that justification alone is insufficient.",
            "Analytic 2": "The principle of charity requires that we interpret others' arguments in their strongest form.",
        }
        
        results = []
        
        for label, text in sample_texts.items():
            prediction = self.classifier.predict(text)
            results.append({
                'Label': label,
                'Text': text[:80] + "..." if len(text) > 80 else text,
                'Continental %': f"{prediction['Continental']*100:.1f}%",
                'Analytic %': f"{prediction['Analytic']*100:.1f}%",
                'Predicted': 'Continental' if prediction['Continental'] > 0.5 else 'Analytic'
            })
        
        # Display as a DataFrame
        df_results = pd.DataFrame(results)
        
        # Style the DataFrame
        def highlight_prediction(row):
            if 'Continental' in row['Label']:
                expected = 'Continental'
            else:
                expected = 'Analytic'
            
            if row['Predicted'] == expected:
                return ['background-color: lightgreen'] * len(row)
            else:
                return ['background-color: lightcoral'] * len(row)
        
        styled_df = df_results.style.apply(highlight_prediction, axis=1)
        display(styled_df)
    
    def create_philosophy_explorer(self):
        """Create an interactive philosophy style explorer."""
        
        # Predefined example texts
        examples = {
            "Select an example...": "",
            "Heidegger - Being and Time": "Dasein is that entity which is in each case mine, and which has, as its manner of Being, the possibility of existing authentically or inauthentically.",
            "Sartre - Existentialism": "Man is condemned to be free; because once thrown into the world, he is responsible for everything he does.",
            "Derrida - Deconstruction": "There is nothing outside the text. Every signified becomes a signifier, in turn deferring to other signifiers in an endless chain.",
            "Russell - Logic": "The fundamental principle in the analysis of propositions containing descriptions is this: Every proposition which we can understand must be composed wholly of constituents with which we are acquainted.",
            "Quine - Ontology": "To be assumed as an entity is, purely and simply, to be reckoned as the value of a variable.",
            "Kripke - Naming": "Proper names are rigid designators, for although the man (Nixon) might not have been the President, it is not the case that he might not have been Nixon.",
        }
        
        # Example selector
        example_dropdown = widgets.Dropdown(
            options=list(examples.keys()),
            value="Select an example...",
            description='Examples:',
            disabled=False,
        )
        
        # Text area
        text_area = widgets.Textarea(
            value='',
            placeholder='Enter philosophical text or select an example above...',
            description='Text:',
            disabled=False,
            layout=widgets.Layout(width='90%', height='120px')
        )
        
        # Analyze button
        analyze_button = widgets.Button(
            description='Analyze Text',
            disabled=False,
            button_style='info',
            tooltip='Analyze the philosophical style',
            icon='search'
        )
        
        # Output
        output = widgets.Output()
        
        def on_example_change(change):
            if change['new'] != "Select an example...":
                text_area.value = examples[change['new']]
        
        def on_analyze_click(b):
            with output:
                clear_output()
                if not self.model_loaded:
                    print("❌ Model not loaded. Please load a trained model first.")
                    return
                
                text = text_area.value.strip()
                if not text:
                    print("❌ Please enter some text to analyze.")
                    return
                
                try:
                    result = self.classifier.predict(text)
                    self.display_detailed_analysis(result, text)
                except Exception as e:
                    print(f"❌ Error during analysis: {str(e)}")
        
        example_dropdown.observe(on_example_change, names='value')
        analyze_button.on_click(on_analyze_click)
        
        # Display
        display(widgets.VBox([
            widgets.HTML("<h3>🔍 Philosophy Style Explorer</h3>"),
            example_dropdown,
            text_area,
            analyze_button
        ]))
        display(output)
    
    def display_detailed_analysis(self, result: Dict[str, float], text: str):
        """Display detailed analysis with style characteristics."""
        
        continental_prob = result['Continental'] * 100
        analytic_prob = result['Analytic'] * 100
        predicted_class = 'Continental' if continental_prob > analytic_prob else 'Analytic'
        
        # Style characteristics
        continental_features = [
            "🏛️ Emphasis on lived experience",
            "🌊 Dialectical thinking",
            "📖 Historical consciousness",
            "🎭 Phenomenological approach",
            "🌀 Existential themes",
            "🔄 Hermeneutical method"
        ]
        
        analytic_features = [
            "🔬 Logical rigor",
            "🎯 Conceptual clarity",
            "⚖️ Formal analysis",
            "🧮 Problem-solving approach",
            "📊 Empirical grounding",
            "🔍 Linguistic precision"
        ]
        
        # Create detailed HTML
        features_html = ""
        if predicted_class == 'Continental':
            features_html = "<br>".join(continental_features)
            color_scheme = "#ff6b6b"
        else:
            features_html = "<br>".join(analytic_features)
            color_scheme = "#4ecdc4"
        
        html_output = f"""
        <div style="border: 2px solid {color_scheme}; border-radius: 15px; padding: 25px; margin: 15px 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
            <h2 style="color: {color_scheme}; margin-top: 0; text-align: center;">📊 Detailed Philosophical Analysis</h2>
            
            <div style="background-color: white; padding: 15px; border-radius: 10px; margin: 15px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="color: #333; margin-top: 0;">📝 Analyzed Text:</h4>
                <p style="font-style: italic; color: #555; line-height: 1.6;">{text}</p>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color: #ff6b6b; margin-top: 0;">🏛️ Continental Philosophy</h4>
                    <div style="font-size: 2em; font-weight: bold; color: #ff6b6b; text-align: center; margin: 10px 0;">
                        {continental_prob:.1f}%
                    </div>
                    <div style="background-color: #f0f0f0; border-radius: 10px; height: 10px; margin: 10px 0;">
                        <div style="background-color: #ff6b6b; height: 100%; border-radius: 10px; width: {continental_prob}%;"></div>
                    </div>
                </div>
                
                <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color: #4ecdc4; margin-top: 0;">🔬 Analytic Philosophy</h4>
                    <div style="font-size: 2em; font-weight: bold; color: #4ecdc4; text-align: center; margin: 10px 0;">
                        {analytic_prob:.1f}%
                    </div>
                    <div style="background-color: #f0f0f0; border-radius: 10px; height: 10px; margin: 10px 0;">
                        <div style="background-color: #4ecdc4; height: 100%; border-radius: 10px; width: {analytic_prob}%;"></div>
                    </div>
                </div>
            </div>
            
            <div style="background-color: white; padding: 20px; border-radius: 10px; margin: 15px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-left: 5px solid {color_scheme};">
                <h4 style="color: {color_scheme}; margin-top: 0;">🎯 Predicted Style: {predicted_class}</h4>
                <p style="color: #666; margin: 5px 0;">Confidence: <strong>{max(continental_prob, analytic_prob):.1f}%</strong></p>
                <h5 style="color: #333; margin-top: 15px;">Characteristic Features:</h5>
                <div style="color: #555; line-height: 1.8;">{features_html}</div>
            </div>
        </div>
        """
        
        display(HTML(html_output))

# Convenience functions for notebook use
def create_interactive_classifier():
    """Create and return an interactive classifier instance."""
    return InteractivePhilosophyClassifier()

def quick_setup():
    """Quick setup function for Jupyter notebooks."""
    classifier = InteractivePhilosophyClassifier()
    
    print("🎓 BERT Philosophical Text Classifier")
    print("=" * 50)
    print()
    
    # Try to load existing model
    if classifier.load_model_interactive():
        print()
        print("🚀 Classifier is ready! You can now:")
        print("   • Use classifier.create_text_input_widget() for interactive classification")
        print("   • Use classifier.create_philosophy_explorer() for detailed analysis")
        print("   • Use classifier.analyze_sample_texts() to see example predictions")
    else:
        print()
        print("🔧 No trained model found. You can:")
        print("   • Use classifier.create_training_widget() to train a new model")
        print("   • Train a model using main.py first")
    
    return classifier
