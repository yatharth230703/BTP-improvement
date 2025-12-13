import modal
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
import os 
# Reuse your existing classes
from src.models.trimodal import TrimodalNetwork
from src.utils.backtester import VectorizedBacktester # Import the new class

# --- Config ---
APP_NAME = "fintech-backtest"
VOLUME_NAME = "fintech-data-vol"
REMOTE_DATA_PATH = Path("/vol/processed")
REMOTE_MODEL_PATH = Path("/vol/models")
REMOTE_PLOT_PATH = Path("/vol/plots")
LOCAL_SRC_DIR = Path("src")

# --- Image ---
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10")
    .pip_install("torch", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "pillow", "torchvision")
    .add_local_python_source(str(LOCAL_SRC_DIR))
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(
    image=image,
    gpu="H100", # Using GPU for fast inference
    volumes={"/vol": volume}
)
def run_backtest_remote(target_stocks):
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image

    # --- Re-Define Dataset Class if not in src ---
    # (Pasting the robust version here to ensure it runs standalone)
    class TrimodalDataset(torch.utils.data.Dataset):
        def __init__(self, df, img_dir, sequence_length=60, target_col='Target_Return'):
            self.seq_len = sequence_length
            drop_cols = [target_col, 'Target_Close', 'Unnamed: 0', 'Date']
            feature_cols = [c for c in df.columns if c not in drop_cols]
            self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)
            self.raw_prices = df['Close'].values # Keep as numpy for index alignment
            self.index_dates = df.index
            
            self.img_tensors = []
            self.valid_indices = []
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_dir_path = Path(img_dir)
            
            # Fast loading loop
            print(f"   ⚡ Loading images for {img_dir}...")
            for i in range(self.seq_len, len(df)):
                date_str = self.index_dates[i].strftime('%Y-%m-%d')
                img_path = img_dir_path / f"{date_str}.png"
                if not img_path.exists(): img_path = img_dir_path / f"{date_str}_gaf.png"
                
                if img_path.exists():
                    try:
                        with Image.open(img_path).convert('RGB') as img:
                            self.img_tensors.append(transform(img))
                            self.valid_indices.append(i)
                    except: pass
            
            if self.img_tensors:
                self.img_tensors = torch.stack(self.img_tensors)

        def __len__(self): return len(self.valid_indices)

        def __getitem__(self, idx):
            i = self.valid_indices[idx]
            return (self.features[i-self.seq_len:i], 
                    self.img_tensors[idx], 
                    self.raw_prices[i-1]) # Return Price_t for reference

        def get_input_dim(self): return self.features.shape[1]

    # --- Main Backtest Loop ---
    for ticker in target_stocks:
        print(f"\n📈 Backtesting Strategy: {ticker}")
        
        # Paths
        csv_path = REMOTE_DATA_PATH / ticker / "full_cleaned.csv"
        img_dir = REMOTE_PLOT_PATH / ticker
        model_path = REMOTE_MODEL_PATH / f"{ticker}_trimodal.pth"
        
        if not model_path.exists():
            print(f"❌ Model not found for {ticker}")
            continue

        # Load Data
        df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        
        # Create Log Returns for Target (needed for scaling fit)
        df['Target_Return'] = np.log((df['Close'] / df['Close'].shift(1)).replace(0, np.nan)).shift(-1)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Split & Scale (Must match training logic exactly)
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        # We only backtest on the TEST set (Unseen data)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size + val_size :].copy()
        
        target_scaler = RobustScaler()
        target_scaler.fit(train_df[['Target_Return']]) # Fit on train
        
        # Dataset
        test_ds = TrimodalDataset(test_df, str(img_dir), sequence_length=60)
        test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
        
        # Load Model
        model = TrimodalNetwork(num_input_dim=test_ds.get_input_dim()).cuda()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Inference
        all_preds_scaled = []
        all_dates = []
        all_actual_prices = []
        
        with torch.no_grad():
            for x_num, x_img, price_t in test_loader:
                x_num, x_img = x_num.cuda(), x_img.cuda()
                preds, _ = model(x_num, x_img)
                all_preds_scaled.extend(preds.cpu().numpy().flatten())
                all_actual_prices.extend(price_t.numpy())
                
        # Inverse Scale Predictions
        pred_returns = target_scaler.inverse_transform(np.array(all_preds_scaled).reshape(-1, 1)).flatten()
        
        # Align Dates (The loader validates indices, we need to map them back)
        valid_dates = test_df.index[test_ds.valid_indices]
        
        # --- RUN BACKTESTER ---
        # Note: We use a small threshold (0.0005) to exploit the "Conservative Bias"
        bt = VectorizedBacktester(ticker, all_actual_prices, pred_returns, valid_dates)
        bt.run_strategy(threshold=0.0005) 
        bt.plot_equity_curve(str(REMOTE_MODEL_PATH))

@app.local_entrypoint()
def main():
    TARGETS = ['SBIN', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'PNB']
    run_backtest_remote.remote(TARGETS)
    
    # Auto-download results
    print("\n⬇️ Downloading Backtest Results...")
    os.makedirs("backtest_results", exist_ok=True)
    
    # CLI-style download for the generated plots
    for t in TARGETS:
        try:
            cmd = f"modal volume get {VOLUME_NAME} /models/{t}_backtest.png backtest_results/{t}_backtest.png"
            os.system(cmd)
            print(f"✅ Downloaded {t}_backtest.png")
        except:
            print(f"❌ Failed to download {t}")