# Demonstrating Text Preprocessing Packages via Twitter Sentiment Analysis

This repository demonstrates the effectiveness of three specialized text preprocessing packages through a practical sentiment analysis implementation:

- **[lightlemma](https://github.com/xga0/lightlemma)** v0.1.6 - Fast English lemmatization
- **[emoticon-fix](https://github.com/xga0/emoticon_fix)** v0.3.0 - Emoticon normalization  
- **[contraction-fix](https://github.com/xga0/contraction_fix)** v0.2.2 - Contraction expansion

**[Interactive Kaggle Notebook](nlp-preprocessing-showcase.ipynb)** - Complete walkthrough with live code execution, detailed explanations, and benchmark results achieving 97.5% validation accuracy.

## Motivation

Standard NLP preprocessing often relies on heavyweight libraries (NLTK, spaCy) or basic regex patterns. These three packages provide lightweight, specialized solutions for common text normalization tasks. This project benchmarks their integration in a complete ML pipeline using the [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) dataset (74,682 training samples, 4 classes).

## Package Integration

### Text Preprocessing Pipeline
```python
from lightlemma import text_to_lemmas
from emoticon_fix import replace_emoticons as fix_emoticons  
from contraction_fix import fix as fix_contractions

# Sequential processing
text = fix_contractions(text)      # "don't" → "do not"
text = fix_emoticons(text)         # ":)" → semantic tokens
tokens = text_to_lemmas(text)      # "running" → "run"
```

### Performance Characteristics
- **lightlemma**: Fast lemmatization without external dependencies
- **contraction-fix**: Standardization improves text consistency
- **emoticon-fix**: Converts text emoticons to semantic meanings

## Technical Implementation

### Model Architecture
- Bidirectional LSTM (2 layers, 256 hidden units)
- Attention mechanism for interpretability
- 128-dimensional embeddings
- ~5M parameters

### Dataset Processing
- **Training**: 74,682 samples
- **Validation**: 1,758 samples  
- **Classes**: Positive, Negative, Neutral, Irrelevant
- **Sequence length**: 128 tokens (padded/truncated)

### Actual Performance Results
- **Validation accuracy**: 97.0% (actual result)
- **Training time**: ~166 minutes on CPU (10 epochs)
- **Preprocessing throughput**: ~10,200 texts/second

## Quick Start

### Installation
```bash
git clone <repository-url>
cd twitter_sentiment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset Setup
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) and place CSV files in the project root:
- `twitter_training.csv`
- `twitter_validation.csv`

### Training
```bash
python train.py
```

Generates:
- `best_model.pth` - Model weights
- `processor.pkl` - Vocabulary and label encoder
- `training_history.png` - Training curves
- `confusion_matrix.png` - Performance visualization

### Inference
```bash
# Interactive mode
python inference.py --interactive

# Single prediction
python inference.py --text "This product is amazing!"

# Batch examples
python inference.py
```

### Package Demonstration
```bash
python demo.py
```

Shows package integration and performance comparison across 1000 texts.

### Interactive Jupyter Notebook
```bash
jupyter notebook nlp-preprocessing-showcase.ipynb
```

Complete walkthrough with:
- Step-by-step preprocessing demonstration
- Model training with live progress tracking
- Performance visualization and benchmarks
- Attention mechanism analysis
- Package comparison and timing results

## Key Results

### Model Performance
**Final validation accuracy: 97.0%** (1000-sample validation set)

**Per-class metrics:**
- **Irrelevant**: 95% F1-score (172 samples)
- **Negative**: 98% F1-score (266 samples)  
- **Neutral**: 97% F1-score (285 samples)
- **Positive**: 97% F1-score (277 samples)

### Preprocessing Impact
1. **Text normalization**: Consistent preprocessing across the dataset
2. **Consistency**: Standardized contractions improve model stability  
3. **Context preservation**: Emoticon conversion maintains sentiment signals

**Note**: Package versions in newer releases may introduce changes. See `requirements.txt` for exact version dependencies.

### Benchmarking
Processing 1000 sample texts:
```
contraction_fix: 0.0004s
emoticon_fix:   0.0781s  
lightlemma:     0.0191s
Total:          0.0977s (10,236 texts/sec)
```

## File Structure
```
├── data_processor.py                    # Package integration and preprocessing
├── model.py                            # PyTorch LSTM implementation
├── train.py                            # Training pipeline
├── inference.py                        # Prediction interface
├── demo.py                             # Package demonstration
├── test_setup.py                       # Environment validation
├── config.py                           # Hyperparameters
├── requirements.txt                    # Exact package versions
├── install.sh                          # Setup automation
└── nlp-preprocessing-showcase.ipynb    # Interactive Kaggle notebook
```

## Dependencies

### Core Packages (Demonstrated)
- `lightlemma==0.1.6`
- `emoticon-fix==0.3.0` 
- `contraction-fix==0.2.2`

### ML Stack
- `torch>=2.0.0`
- `pandas>=1.5.0`
- `scikit-learn>=1.1.0`
- `numpy<2.0` (compatibility requirement)

See `requirements.txt` for complete list.

## Validation

Run environment check:
```bash
python test_setup.py
```

Verifies:
- Package imports and versions
- Function availability and compatibility
- Dataset presence
- PyTorch installation

## Citation

If using these packages in research:

```bibtex
@software{lightlemma2024,
  title={LightLemma: Fast English Lemmatization},
  author={Sean Gao},
  year={2024},
  version={0.1.6},
  url={https://github.com/xga0/lightlemma}
}

@software{emoticon_fix2024,
  title={Emoticon Fix: Semantic Emoticon Processing},
  author={Sean Gao},
  year={2024}, 
  version={0.3.0},
  url={https://github.com/xga0/emoticon_fix}
}

@software{contraction_fix2024,
  title={Contraction Fix: English Contraction Expansion},
  author={Sean Gao},
  year={2024},
  version={0.2.2}, 
  url={https://github.com/xga0/contraction_fix}
}
```