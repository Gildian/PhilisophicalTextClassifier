# Token Analysis Feature - Philosophical Text Classifier

This feature provides detailed token-level analysis to understand how individual words and tokens contribute to the classification decision between Continental and Analytic philosophy styles.

## 🎯 What is Token Analysis?

Token analysis helps you understand:
- **Which specific words** influence the model's decision most
- **How confident** the model is about each token's contribution
- **Visual patterns** in philosophical text classification
- **Attention patterns** that reveal linguistic features the model focuses on

## 🚀 Quick Start

### Option 1: Use the Enhanced Training Script
Train your model with built-in token analysis:

```bash
# Train with token analysis (analyzes 10 sample texts by default)
python train_with_tokens.py --data sample_data.csv --epochs 3

# Analyze more samples during training
python train_with_tokens.py --data sample_data.csv --analysis-samples 20

# Train without token analysis
python train_with_tokens.py --data sample_data.csv --no-token-analysis
```

### Option 2: Run Token Analysis on Existing Model
If you already have a trained model:

```bash
# Run demo with sample texts
python demo_token_analysis.py

# This will analyze 5 sample texts and create visualizations
```

### Option 3: Programmatic Usage
Use the token analysis in your own scripts:

```python
from main import PhilosophyClassifier
from simple_token_analysis import SimpleTokenAnalyzer

# Load your trained model
classifier = PhilosophyClassifier()
classifier.load_model("philosophy_bert_model.pth")

# Create analyzer
analyzer = SimpleTokenAnalyzer(classifier.model, classifier.tokenizer)

# Analyze a text
text = "Being-in-the-world reveals the fundamental structure of existence."
result = analyzer.analyze_token_contributions(text)

# Create visualization
fig = analyzer.create_token_visualization(result, "my_analysis.png")
```

## 📊 Output Files and Visualizations

When you run token analysis, it creates several types of output:

### 1. Individual Token Analysis Plots
- **File**: `token_analysis_*.png`
- **Shows**: Token-by-token importance scores, class probabilities, top contributing tokens, and heatmap visualization
- **Use**: Understand how specific words influence individual predictions

### 2. Summary Analysis Plot
- **File**: `token_summary.png`
- **Shows**: Overall accuracy, confidence distribution, and most important tokens for each philosophy class
- **Use**: Get a bird's-eye view of model behavior across multiple texts

### 3. Analysis Results CSV
- **File**: `analysis_results.csv`
- **Contains**: Predictions, confidence scores, accuracy, and summary statistics
- **Use**: Quantitative analysis and further data processing

### 4. Detailed HTML Report (with full token_analysis.py)
- **File**: `token_analysis_report.html`
- **Contains**: Interactive visualizations and comprehensive analysis
- **Use**: Share results or create presentations

## 🔍 Understanding the Visualizations

### Token Importance Bars
- **Height**: How much each token contributes to the final decision
- **Color**: Red for Continental, Blue for Analytic predictions
- **Labels**: Top contributing tokens are labeled with their importance scores

### Class Probability Charts
- **Shows**: Model's confidence in Continental vs Analytic classification
- **Percentages**: Raw probability scores for each class

### Token Heatmap
- **Grid Layout**: Tokens arranged in a grid pattern
- **Color Intensity**: Darker colors indicate higher importance
- **Pattern Recognition**: Helps identify clusters of important tokens

### Top Contributing Tokens
- **Horizontal Bars**: Most influential tokens ranked by importance
- **Philosophy-Specific**: Separate lists for Continental and Analytic predictions

## 📈 Interpreting Results

### High-Importance Continental Tokens Often Include:
- Existential terms: "being", "existence", "dasein"
- Phenomenological language: "experience", "consciousness", "lived"
- Dialectical concepts: "negation", "contradiction", "becoming"
- Temporal concepts: "history", "time", "historical"

### High-Importance Analytic Tokens Often Include:
- Logical terms: "if", "then", "therefore", "entails"
- Epistemological concepts: "knowledge", "belief", "justified"
- Formal language: "proposition", "argument", "valid"
- Clarity indicators: "precise", "clear", "definition"

### Confidence Levels:
- **90%+**: Very high confidence (model is very certain)
- **80-89%**: High confidence (reliable prediction)
- **70-79%**: Moderate confidence (decent prediction)
- **60-69%**: Low confidence (uncertain prediction)
- **<60%**: Very low confidence (unreliable prediction)

## 🛠️ Technical Details

### How Token Importance is Calculated:
1. **Gradient-Based Method**: Computes gradients of the loss with respect to input token embeddings
2. **Absolute Values**: Uses absolute gradient values as importance scores
3. **Normalization**: Scores are normalized for better visualization

### Requirements:
- **Core**: matplotlib, seaborn, numpy, pandas
- **Enhanced**: plotly, kaleido (for interactive visualizations)
- **Model**: Trained BERT-based philosophical text classifier

### Performance Considerations:
- Token analysis adds computational overhead during training
- Analysis time scales with text length and number of samples
- Visualizations are saved to disk to avoid memory issues

## 🔧 Advanced Usage

### Customizing Analysis:
```python
# Analyze specific texts
custom_texts = ["Your philosophical text here...", "Another text..."]
custom_labels = ["Continental", "Analytic"]

analysis_results, results_df = create_simple_token_analysis(
    classifier, custom_texts, custom_labels, "custom_analysis"
)
```

### Batch Processing:
```python
# Analyze many texts efficiently
analyzer = SimpleTokenAnalyzer(classifier.model, classifier.tokenizer)
batch_df = analyzer.analyze_batch_tokens(text_list, label_list)
```

### Custom Visualizations:
```python
# Create your own plots
result = analyzer.analyze_token_contributions(text)
tokens = result['tokens']
importance = result['token_importance']

# Your custom visualization code here
```

## 🐛 Troubleshooting

### Common Issues:

1. **"Model not found" error**:
   - Train your model first using `python main.py --train`
   - Or use `python train_with_tokens.py --data your_data.csv`

2. **"Token analysis modules not available"**:
   - Install required packages: `pip install -r requirements.txt`
   - Check that `simple_token_analysis.py` is in your directory

3. **Visualizations not showing**:
   - Check that matplotlib backend is properly configured
   - Try saving plots to files instead of showing them

4. **Out of memory errors**:
   - Reduce the number of analysis samples
   - Use smaller batch sizes
   - Analyze texts individually rather than in batches

5. **Poor visualization quality**:
   - Increase DPI in save settings: `plt.savefig('plot.png', dpi=300)`
   - Use vector formats: `plt.savefig('plot.svg')`

### Getting Help:
- Check the console output for detailed error messages
- Ensure your training data format matches the expected CSV/JSON structure
- Verify that your trained model is compatible with the analysis tools

## 📚 Examples and Use Cases

### Academic Research:
- Analyze how different philosophical schools use language
- Study the evolution of philosophical terminology
- Compare model behavior across different text sources

### Model Development:
- Debug classification decisions
- Identify biased or problematic token patterns
- Validate model learning on philosophical concepts

### Educational Applications:
- Teach students about philosophical writing styles
- Demonstrate NLP and attention mechanisms
- Create interactive philosophy learning tools

---

**Happy analyzing! 🎓✨**
