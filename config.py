"""
Configuration file for the Twitter Sentiment Analysis project.
Modify these parameters to customize the model and training process.
"""

# Data Processing Configuration
DATA_CONFIG = {
    'max_length': 128,           # Maximum sequence length
    'use_lemmatization': True,   # Use lemmatization (True) or stemming (False)
    'min_freq': 2,              # Minimum word frequency for vocabulary
    'train_file': 'twitter_training.csv',
    'val_file': 'twitter_validation.csv'
}

# Model Configuration
MODEL_CONFIG = {
    'embedding_dim': 128,        # Word embedding dimension
    'hidden_dim': 256,          # LSTM hidden dimension
    'num_layers': 2,            # Number of LSTM layers
    'dropout': 0.3,             # Dropout rate
    'bidirectional': True,      # Use bidirectional LSTM
    'attention': True,          # Use attention mechanism
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,           # Batch size for training
    'learning_rate': 0.001,     # Learning rate
    'weight_decay': 1e-5,       # Weight decay for regularization
    'num_epochs': 10,           # Number of training epochs
    'early_stopping_patience': 3,  # Early stopping patience
    'use_class_weights': True,  # Use class weights for imbalanced data
    'scheduler_patience': 2,    # Learning rate scheduler patience
    'scheduler_factor': 0.5,    # Learning rate reduction factor
}

# File Paths
PATHS = {
    'model_save_path': 'best_model.pth',
    'processor_save_path': 'processor.pkl',
    'training_plot_path': 'training_history.png',
    'confusion_matrix_path': 'confusion_matrix.png'
}

# Package Versions (for documentation)
PACKAGE_VERSIONS = {
    'lightlemma': '0.1.6',
    'emoticon_fix': '0.3.0',
    'contraction_fix': '0.2.2',
    'torch': '>=2.0.0',
    'torchtext': '>=0.15.0',
    'pandas': '>=1.5.0',
    'numpy': '>=1.21.0',
    'scikit-learn': '>=1.1.0',
    'matplotlib': '>=3.5.0',
    'seaborn': '>=0.11.0',
    'tqdm': '>=4.64.0',
    'transformers': '>=4.20.0'
}

# Text Preprocessing Pipeline Configuration
PREPROCESSING_CONFIG = {
    'remove_urls': True,        # Remove URLs from text
    'remove_mentions': True,    # Remove user mentions (@username)
    'remove_hashtags': False,   # Remove hashtags but keep text
    'remove_numbers': False,    # Remove numbers
    'remove_punctuation': False, # Remove punctuation
    'lowercase': True,          # Convert to lowercase
    'normalize_whitespace': True, # Normalize whitespace
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'save_predictions': True,   # Save predictions to file
    'plot_attention': True,     # Plot attention weights
    'detailed_metrics': True,   # Print detailed evaluation metrics
    'confusion_matrix': True,   # Generate confusion matrix
    'classification_report': True, # Generate classification report
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',        # Logging level
    'save_logs': True,          # Save logs to file
    'log_file': 'training.log', # Log file path
    'print_progress': True,     # Print progress bars
    'verbose': True,            # Verbose output
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'device': 'auto',           # Device to use ('auto', 'cpu', 'cuda')
    'num_workers': 0,           # Number of workers for data loading
    'pin_memory': True,         # Pin memory for faster data transfer
    'mixed_precision': False,   # Use mixed precision training
}

# Model Architecture Options
ARCHITECTURE_OPTIONS = {
    'lstm': {
        'name': 'LSTM with Attention',
        'description': 'Bidirectional LSTM with attention mechanism',
        'default_config': MODEL_CONFIG
    },
    'cnn': {
        'name': 'CNN',
        'description': 'Convolutional Neural Network',
        'config': {
            'embedding_dim': 128,
            'num_filters': 128,
            'filter_sizes': (3, 4, 5),
            'dropout': 0.3
        }
    },
    'transformer': {
        'name': 'Transformer',
        'description': 'Transformer-based model (future implementation)',
        'config': {
            'embedding_dim': 128,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1
        }
    }
}

# Dataset Information
DATASET_INFO = {
    'name': 'Twitter Entity Sentiment Analysis',
    'source': 'https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis',
    'train_samples': 74682,
    'val_samples': 1758,
    'num_classes': 4,
    'classes': ['Positive', 'Negative', 'Neutral', 'Irrelevant'],
    'class_distribution': {
        'Positive': 20832,
        'Negative': 22542,
        'Neutral': 18318,
        'Irrelevant': 12990
    }
}

# Performance Benchmarks (expected)
PERFORMANCE_BENCHMARKS = {
    'accuracy': {
        'target': 0.75,
        'good': 0.80,
        'excellent': 0.85
    },
    'training_time': {
        'cpu': '10-15 minutes',
        'gpu': '3-5 minutes'
    },
    'model_size': '2-3 MB',
    'memory_usage': '2-4 GB RAM'
}
