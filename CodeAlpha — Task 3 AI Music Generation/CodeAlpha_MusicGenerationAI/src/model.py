import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    """
    LSTM-based Language Model for Music Generation.
    
    Architecture Choices:
    - Embedding Dimension: Defaults to 256. This is large enough to capture relationships 
      between ~3500 tokens (pitch and time shifts) without being overly prone to overfitting.
    - LSTM Layers: 3 stacked layers. Music has deep, long-term temporal dependencies 
      (chords, motifs, repeating phrases). 3 layers provide enough depth to learn 
      hierarchical representations of time and melody.
    - Hidden Size: 512. Since we are modeling sequences of 100+ tokens representing 
      complex polyphony, a wide hidden state is necessary to carry context forward.
    - Dropout: 0.3. Prevents overfitting, especially since LSTMs can easily memorize 
      patterns when layered deeply.
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # nn.LSTM throws a warning if dropout > 0 but num_layers == 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Project hidden state back to vocabulary size for next-token prediction
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) containing integer token indices.
            hidden: Tuple of (h_0, c_0), both of shape (num_layers, batch_size, hidden_size).
                    If None, initializes to zeros inside nn.LSTM.
                    
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            hidden: The updated (h_n, c_n) state for passing into the next step.
        """
        # x shape: (batch_size, seq_len)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # hidden: tuple of (h_n, c_n)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        out = self.dropout(lstm_out)
        
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.fc(out)
        
        return logits, hidden
