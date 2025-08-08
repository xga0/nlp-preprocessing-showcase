import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import Tuple, Optional


class TwitterSentimentModel(nn.Module):
    """
    PyTorch model for Twitter sentiment analysis.
    Uses LSTM with attention mechanism for sequence classification.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2, 
                 num_classes: int = 4, dropout: float = 0.3,
                 pretrained_embeddings: Optional[np.ndarray] = None):
        """
        Initialize the model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of sentiment classes
            dropout: Dropout rate
            pretrained_embeddings: Optional pretrained embeddings
        """
        super(TwitterSentimentModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def attention_net(self, lstm_output: torch.Tensor, 
                     final_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention mechanism to weight the importance of each word.
        
        Args:
            lstm_output: Output from LSTM [batch_size, seq_len, hidden_dim*2]
            final_state: Final hidden state from LSTM
            
        Returns:
            Tuple of (context vector, attention weights)
        """
        # lstm_output shape: [batch_size, seq_len, hidden_dim*2]
        # final_state shape: [batch_size, hidden_dim*2]
        
        # Calculate attention weights
        attn_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights
        context = torch.sum(attn_weights * lstm_output, dim=1)  # [batch_size, hidden_dim*2]
        
        return context, attn_weights.squeeze(-1)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input sequences [batch_size, seq_len]
            lengths: Sequence lengths for packing
            
        Returns:
            Logits for each class [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Pack sequences if lengths are provided
        if lengths is not None:
            packed_embedded = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Get final hidden state (concatenate forward and backward)
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Apply attention
        context, attn_weights = self.attention_net(lstm_output, final_hidden)
        
        # Output layers
        out = self.dropout(context)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor, 
                            lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input sequences [batch_size, seq_len]
            lengths: Sequence lengths for packing
            
        Returns:
            Attention weights [batch_size, seq_len]
        """
        batch_size = x.size(0)
        
        # Embedding layer
        embedded = self.embedding(x)
        
        # Pack sequences if lengths are provided
        if lengths is not None:
            packed_embedded = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Get final hidden state
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Get attention weights
        _, attn_weights = self.attention_net(lstm_output, final_hidden)
        
        return attn_weights


class SimpleCNNModel(nn.Module):
    """
    Simple CNN model as an alternative to LSTM.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 num_filters: int = 128, filter_sizes: Tuple[int, ...] = (3, 4, 5),
                 num_classes: int = 4, dropout: float = 0.3):
        """
        Initialize the CNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per filter size
            filter_sizes: Sizes of convolutional filters
            num_classes: Number of sentiment classes
            dropout: Dropout rate
        """
        super(SimpleCNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        
        # Dropout and output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN model.
        
        Args:
            x: Input sequences [batch_size, seq_len]
            
        Returns:
            Logits for each class [batch_size, num_classes]
        """
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, seq_len-k+1, 1]
            conv_out = conv_out.squeeze(3)  # [batch_size, num_filters, seq_len-k+1]
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_out = conv_out.squeeze(2)  # [batch_size, num_filters]
            conv_outputs.append(conv_out)
        
        # Concatenate and classify
        cat = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        cat = self.dropout(cat)
        logits = self.fc(cat)
        
        return logits
