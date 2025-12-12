import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class FinancialDataset(Dataset):
    def __init__(self, csv_path, sequence_length=60, target_col='Close'):
        """
        Args:
            csv_path: Path to the processed .csv file (train/val/test)
            sequence_length: Lookback window (e.g., 60 days)
            target_col: The column we want to predict
        """
        self.df = pd.read_csv(csv_path)
        
        # Drop non-numeric columns (like Date)
        if 'Date' in self.df.columns:
            self.df = self.df.drop(columns=['Date'])
            
        # Ensure all data is float
        self.data = self.df.select_dtypes(include=[np.number]).values
        self.features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Find index of target column for label extraction
        self.target_idx = self.features.index(target_col)
        self.sequence_length = sequence_length

    def __len__(self):
        # We can't use the first 'sequence_length' days
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Window: from idx to idx + sequence_length
        x = self.data[idx : idx + self.sequence_length]
        
        # Label: The value at idx + sequence_length (Next day)
        y = self.data[idx + self.sequence_length, self.target_idx]
        
        # Convert to PyTorch Tensors
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_input_dim(self):
        return self.data.shape[1]