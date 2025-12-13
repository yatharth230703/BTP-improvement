import modal
import os
import sys
from pathlib import Path

# --- Configuration ---
APP_NAME = "fintech-trimodal-research"
VOLUME_NAME = "fintech-data-vol"
REMOTE_MOUNT_PATH = Path("/vol")
REMOTE_DATA_PATH = REMOTE_MOUNT_PATH / "processed"
REMOTE_PLOT_PATH = REMOTE_MOUNT_PATH / "plots"
REMOTE_MODEL_PATH = REMOTE_MOUNT_PATH / "models"
LOCAL_DATA_DIR = Path("data/processed")
LOCAL_PLOT_DIR = Path("data/plots")
LOCAL_SRC_DIR = Path("src")

# --- Define the Modal Image ---
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "torch",
        "pandas",
        "numpy",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "torchvision",
        "pillow",
        "mplfinance",
        "seaborn"
    )
    .add_local_python_source(str(LOCAL_SRC_DIR))
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(volumes={REMOTE_MOUNT_PATH: volume})
def needs_upload():
    return not (REMOTE_DATA_PATH.exists() and REMOTE_PLOT_PATH.exists())

@app.function(
    image=image,
    timeout=1800,
    volumes={REMOTE_MOUNT_PATH: volume},
)
def generate_images_remote(target_stocks):
    import pandas as pd
    from src.utils.ts_image_encoders import generate_ts_images
    
    REMOTE_PLOT_PATH.mkdir(parents=True, exist_ok=True)
    
    for ticker in target_stocks:
        print(f"🖼️ Processing images for {ticker}...")
        stock_dir = REMOTE_DATA_PATH / ticker
        csv_path = stock_dir / "full_cleaned.csv"
        
        if not csv_path.exists(): continue
            
        df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
        ticker_plot_dir = REMOTE_PLOT_PATH / ticker
        ticker_plot_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            generate_ts_images(df, ticker_plot_dir)
            print(f"   ✅ Images generated for {ticker}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={REMOTE_MOUNT_PATH: volume},
)
def train_remote(target_stocks):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd
    from PIL import Image
    from torchvision import transforms
    from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
    from sklearn.preprocessing import RobustScaler
    
    from src.models.trimodal import TrimodalNetwork
    from src.utils.visualizer import Visualizer
    from src.models.trimodal import DirectionalLoss

    # --- DEFINING VISUALIZATION FUNCTION ---
    def visualize_results(ticker, true_prices, pred_prices, save_dir='plots'):
        """
        Generates a 3-panel dashboard to diagnose the Directional Accuracy.
        """
        # Ensure inputs are flat numpy arrays
        if isinstance(true_prices, torch.Tensor):
            true_prices = true_prices.detach().cpu().numpy().flatten()
        if isinstance(pred_prices, torch.Tensor):
            pred_prices = pred_prices.detach().cpu().numpy().flatten()
            
        # Calculate Returns (percentage change) for analysis
        # We use [1:] because we need a previous day to calc return
        # Add epsilon to avoid division by zero
        true_returns = np.diff(true_prices) / (true_prices[:-1] + 1e-9)
        pred_returns = np.diff(pred_prices) / (pred_prices[:-1] + 1e-9)
        
        # Calculate Directions (1 for Up, 0 for Down)
        true_dir = (true_returns > 0).astype(int)
        pred_dir = (pred_returns > 0).astype(int)

        # --- PLOTTING ---
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'Trimodal Model Diagnostics: {ticker}', fontsize=16, fontweight='bold')

        # 1. Price Comparison (Zoomed into last 100 points)
        ax1 = fig.add_subplot(2, 2, 1)
        zoom = 100 
        ax1.plot(true_prices[-zoom:], label='Actual Price', color='blue', linewidth=2, alpha=0.7)
        ax1.plot(pred_prices[-zoom:], label='Predicted Price', color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax1.set_title(f'Price Action Lag Check (Last {zoom} Days)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Directional Confusion Matrix
        ax2 = fig.add_subplot(2, 2, 2)
        cm = confusion_matrix(true_dir, pred_dir)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title(f'Directional Accuracy Confusion Matrix\n(Total Acc: {np.mean(true_dir == pred_dir):.2%})')

        # 3. Returns Scatter Plot
        ax3 = fig.add_subplot(2, 1, 2)
        ax3.scatter(true_returns, pred_returns, alpha=0.5, color='purple')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.axvline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Actual Returns')
        ax3.set_ylabel('Predicted Returns')
        ax3.set_title('Returns Correlation (Target: Top-Right & Bottom-Left Quadrants)')
        ax3.grid(True, alpha=0.3)

        # Quadrant Labels
        if len(true_returns) > 0 and len(pred_returns) > 0:
            ax3.text(max(true_returns)*0.7, max(pred_returns)*0.7, 'Correct UP', color='green', fontweight='bold')
            ax3.text(min(true_returns)*0.7, min(pred_returns)*0.7, 'Correct DOWN', color='green', fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{ticker}_diagnostics.png"
        plt.savefig(save_path)
        print(f"📊 Visualization saved to: {save_path}")
        plt.close()

    print(f"🚀 Starting TRIMODAL Training (Unified Pipeline) on {torch.cuda.get_device_name(0)}")
    
    # --- Dataset Definition ---
    class TrimodalDataset(Dataset):
        def __init__(self, df, img_dir, sequence_length=60, target_col='Target_Return'):
            self.seq_len = sequence_length
            
            # 1. Prepare Numerical Data
            drop_cols = [target_col, 'Target_Close', 'Unnamed: 0', 'Date']
            feature_cols = [c for c in df.columns if c not in drop_cols]
            
            self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)
            # Store raw close prices for reconstruction
            self.raw_prices = torch.tensor(df['Close'].values, dtype=torch.float32)
            self.index_dates = df.index
            
            # 2. Pre-load Images into RAM
            print(f"   ⚡ Pre-loading images into RAM for speed...")
            self.img_tensors = []
            self.valid_indices = []
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_dir_path = Path(img_dir)
            
            for i in range(self.seq_len, len(df)):
                date_str = self.index_dates[i].strftime('%Y-%m-%d')
                img_path = img_dir_path / f"{date_str}.png"
                if not img_path.exists():
                    img_path = img_dir_path / f"{date_str}_gaf.png"
                
                if img_path.exists():
                    try:
                        with Image.open(img_path).convert('RGB') as img:
                            tensor_img = transform(img)
                            self.img_tensors.append(tensor_img)
                            self.valid_indices.append(i)
                    except Exception as e:
                        pass 

            if len(self.img_tensors) > 0:
                self.img_tensors = torch.stack(self.img_tensors)
                print(f"   ✅ Loaded {len(self.img_tensors)} images into RAM")
            else:
                print("   ❌ No images loaded!")

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            i = self.valid_indices[idx]
            
            x_num = self.features[i-self.seq_len : i]
            x_img = self.img_tensors[idx]
            y = self.targets[i] # This is now Log Return
            
            # Pass current price (Price at time t) for reconstruction
            # i is the target index (t+1), so i-1 is current time t
            current_price = self.raw_prices[i-1] 
            
            return x_num, x_img, y, current_price

        def get_input_dim(self):
            return self.features.shape[1]

    # --- Training Config ---
    BATCH_SIZE = 1024
    EPOCHS = 30
    LR = 0.0005
    SEQ_LEN = 60
    REMOTE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    viz = Visualizer(save_dir=str(REMOTE_MODEL_PATH))
    results = {}

    for ticker in target_stocks:
        print(f"\n{'='*60}")
        print(f"👁️ TRIMODAL TRAINING + RECONSTRUCTION (LOG RETURNS): {ticker}")
        print(f"{'='*60}")
        
        stock_dir = REMOTE_DATA_PATH / ticker
        csv_path = stock_dir / "full_cleaned.csv"
        img_dir = REMOTE_PLOT_PATH / ticker
        
        if not csv_path.exists(): continue
            
        # 1. PRE-PROCESSING
        df = pd.read_csv(csv_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        # --- NEW TARGET: LOG RETURNS (WITH FIX) ---
        # Formula: ln(Price_{t+1} / Price_t)
        # Shift(-1) aligns row t with return at t+1
        # ADDED: np.maximum(..., 1e-9) protects against log(0)
        df['Target_Return'] = np.log((df['Close'] / df['Close'].shift(1)).replace(0, np.nan)).shift(-1)
        
        # FIX: Replace infinities with NaN and drop them before scaling
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True) 
        
        # 2. TIME-SERIES SPLIT
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size : train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size :].copy()
        
        # 3. ROBUST SCALING
        target_scaler = RobustScaler()
        train_df['Target_Return'] = target_scaler.fit_transform(train_df[['Target_Return']])
        val_df['Target_Return'] = target_scaler.transform(val_df[['Target_Return']])
        test_df['Target_Return'] = target_scaler.transform(test_df[['Target_Return']])
        
        # Initialize Datasets
        train_ds = TrimodalDataset(train_df, str(img_dir), SEQ_LEN)
        val_ds = TrimodalDataset(val_df, str(img_dir), SEQ_LEN)
        test_ds = TrimodalDataset(test_df, str(img_dir), SEQ_LEN)
        
        if len(train_ds) == 0:
            print("❌ No aligned data. Check image generation.")
            continue
            
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Model Init
        model = TrimodalNetwork(num_input_dim=train_ds.get_input_dim()).cuda()
        criterion = DirectionalLoss(lambda_reg=1.0)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        # --- TRAINING LOOP ---
        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            
            for x_num, x_img, y, _ in train_loader: # Unpack current_price but ignore for training
                x_num, x_img, y = x_num.cuda(), x_img.cuda(), y.float().cuda().unsqueeze(1)
                optimizer.zero_grad()
                preds, gates = model(x_num, x_img)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                
            # Validation Step
            model.eval()
            val_losses = []
            all_val_preds, all_val_targets = [], []
            last_gates = None
            
            with torch.no_grad():
                for x_num, x_img, y, _ in val_loader:
                    x_num, x_img, y = x_num.cuda(), x_img.cuda(), y.float().cuda().unsqueeze(1)
                    preds, gates = model(x_num, x_img)
                    val_losses.append(criterion(preds, y).item())
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_targets.extend(y.cpu().numpy())
                    last_gates = gates.cpu().numpy()

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            r2 = r2_score(all_val_targets, all_val_preds)
            
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train:.5f} | Val: {avg_val:.5f} | R2 (Returns): {r2:.4f}")
            viz.log_step(epoch, avg_train, avg_val, r2, last_gates)

        # Save Model & Gate Plots
        torch.save(model.state_dict(), REMOTE_MODEL_PATH / f"{ticker}_trimodal.pth")
        viz.save_reports(ticker)
        
        # --- 4. PRICE RECONSTRUCTION & METRICS ---
        print("📊 Final Evaluation: Reconstructing Price from Log Returns...")
        model.eval()
        
        all_pred_prices = []
        all_true_prices = []
        
        with torch.no_grad():
            for x_num, x_img, y, current_prices in test_loader:
                x_num, x_img = x_num.cuda(), x_img.cuda()
                current_prices = current_prices.cuda()
                y = y.cuda()
                
                # Get Predicted Log Return (Scaled)
                preds, _ = model(x_num, x_img)
                preds_np = preds.cpu().numpy().flatten()
                
                # Inverse Scale Returns
                pred_log_returns = target_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()
                true_log_returns = target_scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1)).flatten()
                
                # Formula: Price_{t+1} = Price_t * exp(Log_Return)
                current_prices_np = current_prices.cpu().numpy()
                
                batch_pred_prices = current_prices_np * np.exp(pred_log_returns)
                batch_true_prices = current_prices_np * np.exp(true_log_returns)
                
                all_pred_prices.extend(batch_pred_prices)
                all_true_prices.extend(batch_true_prices)

        # Convert to numpy for metrics
        all_pred_prices = np.array(all_pred_prices)
        all_true_prices = np.array(all_true_prices)
        
        # Metrics
        final_r2 = r2_score(all_true_prices, all_pred_prices)
        final_rmse = np.sqrt(mean_squared_error(all_true_prices, all_pred_prices))
        
        # Directional Accuracy (re-calculating returns for precision)
        true_returns_unscaled = target_scaler.inverse_transform(test_df['Target_Return'].values[:len(all_true_prices)].reshape(-1,1)).flatten()
        
        dir_acc = accuracy_score((all_true_prices > (all_true_prices / np.exp(true_returns_unscaled))).astype(int), 
                                 (all_pred_prices > (all_true_prices / np.exp(true_returns_unscaled))).astype(int))

        # --- CALL VISUALIZATION FUNCTION ---
        print(f"📊 Generating Graphs for {ticker}...")
        visualize_results(ticker, all_true_prices, all_pred_prices, save_dir=str(REMOTE_MODEL_PATH))
        
        print(f"✅ {ticker} FINAL RESULTS:")
        print(f"   R2 (Price): {final_r2:.4f}")
        print(f"   RMSE (Price): {final_rmse:.4f}")
        print(f"   Directional Accuracy: {dir_acc:.2%}")
        
        results[ticker] = {"R2": final_r2, "RMSE": final_rmse, "Acc": dir_acc}

    # Final Leaderboard
    print("\n🏆 TRIMODAL LEADERBOARD")
    print(f"{'Ticker':<12} | {'R2':<10} | {'RMSE':<10} | {'Acc':<10}")
    print("-" * 50)
    for t, m in results.items():
        print(f"{t:<12} | {m['R2']:.4f}     | {m['RMSE']:.4f} | {m['Acc']:.2%}")

    return results

@app.local_entrypoint()
def main():
    print("🔄 Verifying Data Sync...")
    vol = modal.Volume.from_name(VOLUME_NAME)
    
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(LOCAL_DATA_DIR, remote_path="processed", recursive=True)
        Path(LOCAL_PLOT_DIR).mkdir(parents=True, exist_ok=True)
        batch.put_directory(LOCAL_PLOT_DIR, remote_path="plots", recursive=True)

    TARGET_STOCKS = ['SBIN', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'PNB']
    
    print("\n🔨 Generating Images...")
    generate_images_remote.remote(TARGET_STOCKS)
    
    print("\n🧠 Training Trimodal Model...")
    train_remote.remote(TARGET_STOCKS)