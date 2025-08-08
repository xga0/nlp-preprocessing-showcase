import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import time
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_processor import TwitterDataProcessor
from model import TwitterSentimentModel, SimpleCNNModel


class Trainer:
    """
    Trainer class for the Twitter sentiment analysis model.
    """
    
    def __init__(self, model, device, train_loader, val_loader, 
                 criterion, optimizer, scheduler=None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, num_epochs, early_stopping_patience=5):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Best validation accuracy
        """
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, predictions, targets = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return best_val_acc
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def create_data_loaders(train_data, val_data, batch_size=32):
    """
    Create PyTorch data loaders.
    
    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TensorDataset(
        torch.LongTensor(train_data['sequences']),
        torch.LongTensor(train_data['labels'])
    )
    
    val_dataset = TensorDataset(
        torch.LongTensor(val_data['sequences']),
        torch.LongTensor(val_data['labels'])
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def evaluate_model(model, val_loader, device, label_encoder):
    """
    Evaluate the model and print detailed metrics.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to evaluate on
        label_encoder: Label encoder for converting indices to labels
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Convert indices to labels
    pred_labels = label_encoder.inverse_transform(all_predictions)
    true_labels = label_encoder.inverse_transform(all_targets)
    
    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(true_labels, pred_labels))
    
    # Print confusion matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(true_labels, pred_labels, labels=label_encoder.classes_)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")


def main():
    """Main training function."""
    print("="*80)
    print("TWITTER SENTIMENT ANALYSIS WITH LIGHTLEMMA, EMOTICON_FIX, AND CONTRACTION_FIX")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize data processor
    print("\nInitializing data processor...")
    processor = TwitterDataProcessor(max_length=128, use_lemmatization=True)
    
    # Load and process data
    print("\nLoading and processing data...")
    train_df, val_df = processor.load_data('twitter_training.csv', 'twitter_validation.csv')
    train_data, val_data = processor.process_data(train_df, val_df)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(train_data, val_data, batch_size=32)
    
    # Calculate class weights for imbalanced dataset
    class_weights = processor.get_class_weights(train_data['labels'])
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class weights: {class_weights}")
    print(f"Class distribution: {np.bincount(train_data['labels'])}")
    
    # Initialize model
    print("\nInitializing model...")
    model = TwitterSentimentModel(
        vocab_size=processor.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=len(processor.label_encoder.classes_),
        dropout=0.3
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Initialize trainer
    trainer = Trainer(model, device, train_loader, val_loader, criterion, optimizer, scheduler)
    
    # Train the model
    print("\nStarting training...")
    start_time = time.time()
    best_val_acc = trainer.train(num_epochs=10, early_stopping_patience=3)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Load best model and evaluate
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, val_loader, device, processor.label_encoder)
    
    # Save processor for inference
    import pickle
    with open('processor.pkl', 'wb') as f:
        pickle.dump(processor, f)
    
    print("\nModel and processor saved successfully!")
    print("Files created:")
    print("- best_model.pth: Trained model weights")
    print("- processor.pkl: Data processor with vocabulary and label encoder")
    print("- training_history.png: Training curves")
    print("- confusion_matrix.png: Confusion matrix")


if __name__ == "__main__":
    main()
