import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import string
from typing import List, Tuple, Dict, Any

# Import the three packages to demonstrate their usefulness
from lightlemma import text_to_lemmas, text_to_stems
from emoticon_fix import replace_emoticons as fix_emoticons
from contraction_fix import fix as fix_contractions


class TwitterDataProcessor:
    """
    Data processor for Twitter sentiment analysis that demonstrates the usefulness
    of lightlemma, emoticon_fix, and contraction_fix packages.
    """
    
    def __init__(self, max_length: int = 128, use_lemmatization: bool = True):
        """
        Initialize the data processor.
        
        Args:
            max_length: Maximum sequence length for tokenization
            use_lemmatization: Whether to use lemmatization (True) or stemming (False)
        """
        self.max_length = max_length
        self.use_lemmatization = use_lemmatization
        self.label_encoder = LabelEncoder()
        self.vocab = {}
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
        
    def load_data(self, train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and validation data.
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            
        Returns:
            Tuple of (train_df, val_df)
        """
        print("Loading data...")
        
        # Load data with proper column names
        train_df = pd.read_csv(train_path, header=None, 
                              names=['id', 'entity', 'sentiment', 'text'])
        val_df = pd.read_csv(val_path, header=None, 
                            names=['id', 'entity', 'sentiment', 'text'])
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Validation data shape: {val_df.shape}")
        
        return train_df, val_df
    
    def clean_text(self, text: str) -> str:
        """
        Clean text using the three packages to demonstrate their usefulness.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # Step 1: Fix contractions using contraction_fix
        # This package handles contractions like "don't" -> "do not"
        text = fix_contractions(text)
        
        # Step 2: Fix emoticons using emoticon_fix
        # This package converts emoticons to their semantic meaning
        text = fix_emoticons(text)
        
        # Step 3: Basic text cleaning
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_text(self, text: str) -> List[str]:
        """
        Normalize text using lightlemma package.
        
        Args:
            text: Cleaned text to normalize
            
        Returns:
            List of normalized tokens
        """
        if not text:
            return []
        
        try:
            # Use lightlemma for text normalization
            if self.use_lemmatization:
                # Use lemmatization for more accurate results
                tokens = text_to_lemmas(text, preserve_original_case=False)
            else:
                # Use stemming for faster processing
                tokens = text_to_stems(text, preserve_original_case=False)
        except ValueError as e:
            if "too long" in str(e):
                # Handle cases where words are too long for the lemmatizer
                # Fall back to simple tokenization and filtering
                import re
                tokens = re.findall(r'\b\w+\b', text.lower())
                # Filter out words longer than 50 characters
                tokens = [token for token in tokens if len(token) <= 50]
            else:
                raise e
        
        return tokens
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2) -> None:
        """
        Build vocabulary from processed texts.
        
        Args:
            texts: List of processed text strings
            min_freq: Minimum frequency for a word to be included in vocabulary
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        word_freq = {}
        for text in texts:
            tokens = self.normalize_text(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Filter by minimum frequency
        vocab_words = [word for word, freq in word_freq.items() if freq >= min_freq]
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        self.vocab = special_tokens + sorted(vocab_words)
        self.vocab_size = len(self.vocab)
        
        # Create word to index mapping
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {vocab_words[:10]}")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of indices.
        
        Args:
            text: Input text
            
        Returns:
            List of token indices
        """
        tokens = self.normalize_text(text)
        
        # Convert tokens to indices
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # Add start and end tokens
        sequence = [self.word2idx['<START>']] + sequence + [self.word2idx['<END>']]
        
        # Pad or truncate to max_length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [self.word2idx['<PAD>']] * (self.max_length - len(sequence))
        
        return sequence
    
    def process_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process the entire dataset.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            
        Returns:
            Tuple of (train_data, val_data) dictionaries
        """
        print("Processing data...")
        
        # Clean texts
        print("Cleaning texts...")
        train_df['cleaned_text'] = train_df['text'].apply(self.clean_text)
        val_df['cleaned_text'] = val_df['text'].apply(self.clean_text)
        
        # Show examples of text cleaning
        print("\n=== Text Cleaning Examples ===")
        for i in range(3):
            original = train_df['text'].iloc[i]
            cleaned = train_df['cleaned_text'].iloc[i]
            print(f"Original: {original}")
            print(f"Cleaned:  {cleaned}")
            print(f"Normalized: {' '.join(self.normalize_text(cleaned))}")
            print("-" * 50)
        
        # Build vocabulary from training data
        self.build_vocabulary(train_df['cleaned_text'].tolist())
        
        # Encode labels
        print("Encoding labels...")
        all_labels = pd.concat([train_df['sentiment'], val_df['sentiment']])
        self.label_encoder.fit(all_labels)
        
        train_labels = self.label_encoder.transform(train_df['sentiment'])
        val_labels = self.label_encoder.transform(val_df['sentiment'])
        
        # Convert texts to sequences
        print("Converting texts to sequences...")
        train_sequences = [self.text_to_sequence(text) for text in train_df['cleaned_text']]
        val_sequences = [self.text_to_sequence(text) for text in val_df['cleaned_text']]
        
        # Prepare data dictionaries
        train_data = {
            'sequences': np.array(train_sequences),
            'labels': np.array(train_labels),
            'texts': train_df['text'].tolist(),
            'cleaned_texts': train_df['cleaned_text'].tolist(),
            'entities': train_df['entity'].tolist()
        }
        
        val_data = {
            'sequences': np.array(val_sequences),
            'labels': np.array(val_labels),
            'texts': val_df['text'].tolist(),
            'cleaned_texts': val_df['cleaned_text'].tolist(),
            'entities': val_df['entity'].tolist()
        }
        
        print(f"Training sequences shape: {train_data['sequences'].shape}")
        print(f"Validation sequences shape: {val_data['sequences'].shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return train_data, val_data
    
    def get_class_weights(self, labels: np.ndarray) -> np.ndarray:
        """
        Calculate class weights for imbalanced dataset.
        
        Args:
            labels: Array of labels
            
        Returns:
            Array of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        return class_weights
