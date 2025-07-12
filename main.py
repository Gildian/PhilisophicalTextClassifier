#!/usr/bin/env python3
"""
BERT-based Philosophical Text Classifier
Classifies text as Continental vs Analytic Philosophy

Author: Gildian Gonzales
Date: July 11, 2025
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
import argparse
from datetime import datetime
import PyPDF2
import re

class PhilosophicalBERTClassifier(nn.Module):
    """
    BERT-based classifier for distinguishing between Continental and Analytic philosophy texts.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2, dropout_rate: float = 0.3):
        super(PhilosophicalBERTClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        probabilities = self.softmax(logits)
        return logits, probabilities

class PhilosophyClassifier:
    """
    Main classifier class that handles training, evaluation, and prediction.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class mapping
        self.class_names = ['Continental', 'Analytic']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        

    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode text for BERT input.
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoding
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Prepare data loaders for training and validation.
        """
        # Convert labels to indices
        df['label_idx'] = df['label'].map(self.class_to_idx)
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        # Create datasets
        train_dataset = PhilosophyDataset(train_df, self.tokenizer, self.max_length)
        val_dataset = PhilosophyDataset(val_df, self.tokenizer, self.max_length)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, epochs: int = 3, learning_rate: float = 2e-5):
        """
        Train the BERT classifier.
        """
        self.model = PhilosophicalBERTClassifier(self.model_name).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                logits, _ = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    logits, probabilities = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    
                    total_val_loss += loss.item()
                    
                    predictions = torch.argmax(probabilities, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_predictions)
            
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print("-" * 50)
        
        return train_losses, val_losses, val_accuracies
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict the philosophical style of a given text.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        self.model.eval()
        
        # Preprocess text
        encoding = self.preprocess_text(text)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits, probabilities = self.model(input_ids, attention_mask)
            probs = probabilities.cpu().numpy()[0]
        
        result = {}
        for idx, prob in enumerate(probs):
            class_name = self.idx_to_class[idx]
            result[class_name] = float(prob)
        
        return result
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'class_names': self.class_names
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = PhilosophicalBERTClassifier(checkpoint['model_name']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_name = checkpoint['model_name']
        self.max_length = checkpoint['max_length']
        self.class_names = checkpoint['class_names']
        
        print(f"Model loaded from {path}")
    
    def load_data_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load philosophical texts from a CSV file.
        CSV should have columns: 'text' and 'label'
        Labels should be either 'Continental' or 'Analytic'
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("CSV must contain 'text' and 'label' columns")
            
            # Validate labels
            valid_labels = set(self.class_names)
            invalid_labels = set(df['label'].unique()) - valid_labels
            if invalid_labels:
                raise ValueError(f"Invalid labels found: {invalid_labels}. Use only: {valid_labels}")
            
            print(f"Loaded {len(df)} samples from {csv_path}")
            print(f"Continental: {len(df[df['label'] == 'Continental'])}")
            print(f"Analytic: {len(df[df['label'] == 'Analytic'])}")
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise
    
    def load_data_from_json(self, json_path: str) -> pd.DataFrame:
        """
        Load philosophical texts from a JSON file.
        JSON should be a list of objects with 'text' and 'label' keys
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            # Validate required columns
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("JSON must contain 'text' and 'label' fields")
            
            # Validate labels
            valid_labels = set(self.class_names)
            invalid_labels = set(df['label'].unique()) - valid_labels
            if invalid_labels:
                raise ValueError(f"Invalid labels found: {invalid_labels}. Use only: {valid_labels}")
            
            print(f"Loaded {len(df)} samples from {json_path}")
            print(f"Continental: {len(df[df['label'] == 'Continental'])}")
            print(f"Analytic: {len(df[df['label'] == 'Analytic'])}")
            
            return df
            
        except Exception as e:
            print(f"Error loading JSON: {e}")
            raise
    
    def load_data_from_pdf(self, pdf_path: str, label: str) -> pd.DataFrame:
        """
        Extract text from a PDF file and create training data.
        
        Args:
            pdf_path: Path to the PDF file
            label: Label for the text ('Continental' or 'Analytic')
        
        Returns:
            DataFrame with extracted text chunks and labels
        """
        try:
            # Extract text from PDF
            text_chunks = self._extract_text_from_pdf(pdf_path)
            
            # Validate label
            if label not in self.class_names:
                raise ValueError(f"Invalid label: {label}. Use only: {self.class_names}")
            
            # Create DataFrame
            data = []
            for chunk in text_chunks:
                if len(chunk.strip()) > 50:  # Only include substantial text chunks
                    data.append({'text': chunk.strip(), 'label': label})
            
            df = pd.DataFrame(data)
            print(f"Extracted {len(df)} text chunks from {pdf_path}")
            print(f"All labeled as: {label}")
            
            return df
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise
    
    def _extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extract text from PDF and split into meaningful chunks.
        """
        try:
            text_chunks = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
            
            # Clean the text
            full_text = self._clean_extracted_text(full_text)
            
            # Split into chunks by sentences/paragraphs
            chunks = self._split_text_into_chunks(full_text)
            
            return chunks
            
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            raise
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted PDF text by removing artifacts and normalizing.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'\n[A-Z\s]{10,}\n', '\n', text)
        
        # Fix common PDF extraction issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        
        return text.strip()
    
    def _split_text_into_chunks(self, text: str, max_length: int = 400) -> List[str]:
        """
        Split text into meaningful chunks for training.
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
                
            # If paragraph is too long, split by sentences
            if len(paragraph) > max_length:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= max_length:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(paragraph)
        
        return chunks

class PhilosophyDataset(torch.utils.data.Dataset):
    """
    Custom dataset for philosophical texts.
    """
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        label = row['label_idx']
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    """
    Main function to demonstrate the philosophical text classifier.
    """
    parser = argparse.ArgumentParser(description='BERT Philosophical Text Classifier')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Text to classify')
    parser.add_argument('--model_path', type=str, default='philosophy_bert_model.pth', help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--data_csv', type=str, help='Path to CSV file with training data')
    parser.add_argument('--data_json', type=str, help='Path to JSON file with training data')
    parser.add_argument('--data_pdf', type=str, help='Path to PDF file with training data')
    parser.add_argument('--pdf_label', type=str, choices=['Continental', 'Analytic'], help='Label for PDF data (Continental or Analytic)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = PhilosophyClassifier()
    
    if args.train:
        # Load data based on provided arguments
        if args.data_csv:
            print(f"Loading data from CSV: {args.data_csv}")
            df = classifier.load_data_from_csv(args.data_csv)
        elif args.data_json:
            print(f"Loading data from JSON: {args.data_json}")
            df = classifier.load_data_from_json(args.data_json)
        elif args.data_pdf:
            if not args.pdf_label:
                print("Error: --pdf_label is required when using --data_pdf")
                print("Please specify either 'Continental' or 'Analytic' for the PDF content")
                return
            print(f"Loading data from PDF: {args.data_pdf} (Label: {args.pdf_label})")
            df = classifier.load_data_from_pdf(args.data_pdf, args.pdf_label)
        else:
            print("Error: No training data provided.")
            print("Please specify one of the following data sources:")
            print("  --data_csv path/to/data.csv")
            print("  --data_json path/to/data.json")
            print("  --data_pdf path/to/data.pdf --pdf_label Continental|Analytic")
            return
        print()
        
        print("Preparing data loaders...")
        train_loader, val_loader = classifier.prepare_data(df)
        
        print("Training model...")
        train_losses, val_losses, val_accuracies = classifier.train_model(
            train_loader, val_loader, epochs=args.epochs
        )
        
        print("Saving model...")
        classifier.save_model(args.model_path)
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("Training curves saved to training_curves.png")
    
    elif args.predict:
        if os.path.exists(args.model_path):
            print("Loading trained model...")
            classifier.load_model(args.model_path)
            
            print(f"Analyzing text: '{args.predict[:100]}{'...' if len(args.predict) > 100 else ''}'")
            print()
            
            result = classifier.predict(args.predict)
            
            print("Classification Results:")
            print("-" * 40)
            for style, probability in result.items():
                percentage = probability * 100
                print(f"{style}: {percentage:.2f}%")
            
            # Determine the predicted class
            predicted_class = max(result, key=result.get)
            confidence = result[predicted_class] * 100
            
            print(f"\nPredicted Style: {predicted_class} (Confidence: {confidence:.2f}%)")
        
        else:
            print(f"Model file {args.model_path} not found. Please train the model first using --train flag.")
    
    else:
        print("Please specify either --train to train the model or --predict 'text' to classify text.")
        print("Example usage:")
        print("  python main.py --train")
        print("  python main.py --train --data_csv data.csv")
        print("  python main.py --train --data_json data.json")
        print("  python main.py --train --data_pdf data.pdf")
        print("  python main.py --predict 'Being-in-the-world reveals the structure of Dasein.'")

if __name__ == "__main__":
    main()