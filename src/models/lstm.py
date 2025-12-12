import torch
import torch.nn as nn

class DualBranchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(DualBranchLSTM, self).__init__()
        
        # Branch 1: Numerical/Price Data Processing
        # We assume input_dim includes all features. The LSTM learns to weigh them.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention Mechanism (Optional but powerful)
        # Allows model to focus on specific days in the 60-day window
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Fully Connected Output Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Features)
        
        # 1. Pass through LSTM
        # out shape: (Batch, Seq_Len, Hidden_Dim)
        # _ (hidden states): ignored for now
        lstm_out, _ = self.lstm(x)
        
        # 2. Extract the last time step (Standard LSTM usage)
        # This represents the summary of the entire 60-day window
        last_step_feature = lstm_out[:, -1, :]
        
        # 3. Prediction
        prediction = self.fc(last_step_feature)
        
        return prediction