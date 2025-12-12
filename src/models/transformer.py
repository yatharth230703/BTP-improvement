import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=2):
        super(TimeSeriesTransformer, self).__init__()
        
        # 1. Embedding Layer (Linear projection to d_model size)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding (To give the model a sense of time order)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head (Classification)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # Output_dim=2 for Binary Classification (Up/Down)
        )

    def forward(self, src):
        # src shape: [batch, seq_len, features]
        
        # Embed and add position info
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        # Pass through Transformer
        # output shape: [batch, seq_len, d_model]
        output = self.transformer_encoder(src)
        
        # We take the LAST time step's output as the summary of the sequence
        last_step_output = output[:, -1, :]
        
        # Prediction
        return self.fc(last_step_output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input embedding
        return x + self.pe[:x.size(1), :]