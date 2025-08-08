import torch
import pickle
import numpy as np
from typing import List, Dict, Any
import argparse

from data_processor import TwitterDataProcessor
from model import TwitterSentimentModel


class TwitterSentimentPredictor:
    """
    Class for making predictions on new Twitter text using the trained model.
    """
    
    def __init__(self, model_path: str, processor_path: str, device: str = 'auto'):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model weights
            processor_path: Path to the saved processor
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        # Load processor
        with open(processor_path, 'rb') as f:
            self.processor = pickle.load(f)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = TwitterSentimentModel(
            vocab_size=self.processor.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_classes=len(self.processor.label_encoder.classes_),
            dropout=0.3
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Available classes: {self.processor.label_encoder.classes_}")
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing prediction results
        """
        # Clean and process text
        cleaned_text = self.processor.clean_text(text)
        sequence = self.processor.text_to_sequence(cleaned_text)
        
        # Convert to tensor
        input_tensor = torch.LongTensor([sequence]).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get class label
        predicted_label = self.processor.label_encoder.inverse_transform([predicted_class])[0]
        
        # Get all class probabilities
        class_probs = {}
        for i, class_name in enumerate(self.processor.label_encoder.classes_):
            class_probs[class_name] = probabilities[0][i].item()
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'predicted_sentiment': predicted_label,
            'confidence': confidence,
            'class_probabilities': class_probs,
            'normalized_tokens': self.processor.normalize_text(cleaned_text)
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        return results
    
    def demonstrate_packages(self, text: str):
        """
        Demonstrate the three packages used in text preprocessing.
        
        Args:
            text: Input text to demonstrate preprocessing
        """
        print("="*80)
        print("DEMONSTRATING THE THREE PACKAGES")
        print("="*80)
        
        print(f"Original text: {text}")
        print("-" * 50)
        
        # Step 1: Show contraction fixing
        from contraction_fix import fix as fix_contractions
        contracted_text = fix_contractions(text)
        print(f"After contraction_fix: {contracted_text}")
        print("-" * 50)
        
        # Step 2: Show emoticon fixing
        from emoticon_fix import replace_emoticons as fix_emoticons
        emoticon_fixed_text = fix_emoticons(contracted_text)
        print(f"After emoticon_fix: {emoticon_fixed_text}")
        print("-" * 50)
        
        # Step 3: Show lemmatization
        from lightlemma import text_to_lemmas, text_to_stems
        lemmas = text_to_lemmas(emoticon_fixed_text)
        stems = text_to_stems(emoticon_fixed_text)
        print(f"After lightlemma (lemmatization): {' '.join(lemmas)}")
        print(f"After lightlemma (stemming): {' '.join(stems)}")
        print("-" * 50)
        
        # Step 4: Show final cleaned text
        final_cleaned = self.processor.clean_text(text)
        print(f"Final cleaned text: {final_cleaned}")
        print("="*80)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Twitter Sentiment Analysis Inference')
    parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    parser.add_argument('--processor', default='processor.pkl', help='Path to processor')
    parser.add_argument('--text', help='Single text to predict')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = TwitterSentimentPredictor(args.model, args.processor)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model first by running 'python train.py'")
        return
    
    # Demonstrate the packages with example text
    example_text = "I don't like this product at all! 😡 It's terrible and I'm not happy with it."
    predictor.demonstrate_packages(example_text)
    
    # Single text prediction
    if args.text:
        result = predictor.predict_sentiment(args.text)
        print(f"\nPrediction for: {result['text']}")
        print(f"Sentiment: {result['predicted_sentiment']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Class probabilities: {result['class_probabilities']}")
    
    # Interactive mode
    elif args.interactive:
        print("\n" + "="*60)
        print("INTERACTIVE MODE - Enter tweets to analyze (type 'quit' to exit)")
        print("="*60)
        
        while True:
            text = input("\nEnter a tweet: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if text:
                result = predictor.predict_sentiment(text)
                print(f"\nSentiment: {result['predicted_sentiment']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Cleaned text: {result['cleaned_text']}")
                print(f"Normalized tokens: {' '.join(result['normalized_tokens'])}")
    
    # Default: show some example predictions
    else:
        print("\n" + "="*60)
        print("EXAMPLE PREDICTIONS")
        print("="*60)
        
        example_texts = [
            "I love this new iPhone! It's amazing and works perfectly! 😍",
            "This service is terrible. I'm so frustrated with the poor quality 😠",
            "The weather is okay today, nothing special.",
            "I don't care about this topic at all, it's irrelevant to me.",
            "Don't buy this product! It's a complete waste of money and time 😡",
            "This restaurant has the best food ever! Highly recommend! ⭐⭐⭐⭐⭐",
            "The movie was decent, not great but not bad either.",
            "I can't believe how good this game is! It's absolutely fantastic! 🎮"
        ]
        
        for text in example_texts:
            result = predictor.predict_sentiment(text)
            print(f"\nText: {result['text']}")
            print(f"Sentiment: {result['predicted_sentiment']} (Confidence: {result['confidence']:.3f})")
            print(f"Cleaned: {result['cleaned_text']}")
            print("-" * 50)


if __name__ == "__main__":
    main()
