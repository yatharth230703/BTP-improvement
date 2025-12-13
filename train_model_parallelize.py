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
    import numpy as np
    import pandas as pd
    from PIL import Image
    from torchvision import transforms
    from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
    from sklearn.preprocessing import RobustScaler
    
    from src.models.trimodal import TrimodalNetwork
    from src.utils.visualizer import Visualizer
    from src.models.trimodal import DirectionalLoss
    from src.utils.visualize_Res import visualize_results
    print(f"🚀 Starting TRIMODAL Training (Unified Pipeline) on {torch.cuda.get_device_name(0)}")
    
    # --- Dataset Definition (Preserving logic from both versions) ---
    # --- Optimized Dataset (Pre-loads to RAM) ---
    class TrimodalDataset(Dataset):
        def __init__(self, df, img_dir, sequence_length=60, target_col='Target_Return'):
            self.seq_len = sequence_length
            
            # 1. Prepare Numerical Data
            drop_cols = [target_col, 'Target_Close', 'Unnamed: 0']
            feature_cols = [c for c in df.columns if c not in drop_cols]
            self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)
            self.index_dates = df.index
            
            # 2. Pre-load Images into RAM (The Speedup Fix)
            print(f"   ⚡ Pre-loading images into RAM for speed...")
            self.img_tensors = []
            self.valid_indices = []
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_dir_path = Path(img_dir)
            
            # Loop once and cache everything
            for i in range(self.seq_len, len(df)):
                date_str = self.index_dates[i].strftime('%Y-%m-%d')
                
                # Check for image existence
                img_path = img_dir_path / f"{date_str}.png"
                if not img_path.exists():
                    img_path = img_dir_path / f"{date_str}_gaf.png"
                
                if img_path.exists():
                    try:
                        # Load and transform immediately
                        with Image.open(img_path).convert('RGB') as img:
                            tensor_img = transform(img)
                            self.img_tensors.append(tensor_img)
                            self.valid_indices.append(i)
                    except Exception as e:
                        pass # Skip corrupt images

            # Stack images into a single massive tensor (N, 3, 224, 224)
            if len(self.img_tensors) > 0:
                self.img_tensors = torch.stack(self.img_tensors)
                print(f"   ✅ Loaded {len(self.img_tensors)} images into RAM (~{self.img_tensors.element_size() * self.img_tensors.numel() / 1024**2:.2f} MB)")
            else:
                print("   ❌ No images loaded!")

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            # No disk I/O here! Just array slicing.
            i = self.valid_indices[idx]
            
            # Numerical Input
            x_num = self.features[i-self.seq_len : i]
            
            # Image Input (Already in RAM)
            x_img = self.img_tensors[idx]
            
            # Target
            y = self.targets[i]
            
            return x_num, x_img, y

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
        print(f"👁️ TRIMODAL TRAINING + RECONSTRUCTION: {ticker}")
        print(f"{'='*60}")
        
        stock_dir = REMOTE_DATA_PATH / ticker
        csv_path = stock_dir / "full_cleaned.csv"
        img_dir = REMOTE_PLOT_PATH / ticker
        
        if not csv_path.exists(): continue
            
        # 1. ROBUST PRE-PROCESSING (From Old Version)
        df = pd.read_csv(csv_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        # Recalculate Log Returns
        price_ratio = df['Close'].shift(-1) / df['Close']
        valid_mask = (price_ratio > 0) & (price_ratio.notna())
        df['Target_Return'] = 0.0
        df.loc[valid_mask, 'Target_Return'] = np.log(price_ratio[valid_mask])
        df['Target_Return'] = df['Target_Return'].replace([np.inf, -np.inf], 0.0)
        df.dropna(inplace=True)
        
        # 2. TIME-SERIES SPLIT (From Old Version - No Random Split!)
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size : train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size :].copy()
        
        # 3. ROBUST SCALING (From Old Version)
        target_scaler = RobustScaler()
        train_df['Target_Return'] = target_scaler.fit_transform(train_df[['Target_Return']])
        val_df['Target_Return'] = target_scaler.transform(val_df[['Target_Return']])
        test_df['Target_Return'] = target_scaler.transform(test_df[['Target_Return']])
        
        # Initialize Datasets with Split DataFrames
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
            
            for x_num, x_img, y in train_loader:
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
                for x_num, x_img, y in val_loader:
                    x_num, x_img, y = x_num.cuda(), x_img.cuda(), y.float().cuda().unsqueeze(1)
                    preds, gates = model(x_num, x_img)
                    val_losses.append(criterion(preds, y).item())
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_targets.extend(y.cpu().numpy())
                    last_gates = gates.cpu().numpy()

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            r2 = r2_score(all_val_targets, all_val_preds)
            
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train:.5f} | Val: {avg_val:.5f} | R2: {r2:.4f}")
            viz.log_step(epoch, avg_train, avg_val, r2, last_gates)

        # Save Model & Gate Plots
        torch.save(model.state_dict(), REMOTE_MODEL_PATH / f"{ticker}_trimodal.pth")
        viz.save_reports(ticker)
        
        # --- 4. PRICE RECONSTRUCTION & METRICS (From Old Version) ---
        print("📊 Final Evaluation: Reconstructing Price...")
        model.eval()
        
        val_predictions = []
        val_targets = []
        pred_returns_scaled = []
        with torch.no_grad():
            for x_num, x_img, _ in test_loader:
                x_num, x_img = x_num.cuda(), x_img.cuda()
                preds, _ = model(x_num, x_img)
                pred_returns_scaled.extend(preds.cpu().numpy().flatten())
                val_predictions.append(outputs.cpu())
                val_targets.append(targets.cpu())
        val_predictions = torch.cat(val_predictions).numpy()
        val_targets = torch.cat(val_targets).numpy()
        # Inverse Scale Returns
        pred_log_returns = target_scaler.inverse_transform(np.array(pred_returns_scaled).reshape(-1, 1)).flatten()
        
        # Align Prices
        # We need the 'Close' prices corresponding to the start of the prediction interval
        # The test_loader iterates over valid indices. We need to map those back to df.
        test_indices = test_ds.valid_indices
        base_prices = test_df['Close'].iloc[[i-1 for i in test_indices]].values # Price at t
        actual_future_prices = test_df['Close'].iloc[test_indices].values       # Price at t+1
        
        # Apply Returns: P_t+1 = P_t * exp(return)
        predicted_prices = base_prices * np.exp(pred_log_returns)
        
        # Metrics
        final_r2 = r2_score(actual_future_prices, predicted_prices)
        final_rmse = np.sqrt(mean_squared_error(actual_future_prices, predicted_prices))
        
        # Directional Accuracy
        actual_move = np.sign(actual_future_prices - base_prices)
        pred_move = np.sign(predicted_prices - base_prices)
        dir_acc = accuracy_score(actual_move, pred_move)
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(actual_future_prices, label='Actual', alpha=0.6, color='black')
        plt.plot(predicted_prices, label='Predicted (Trimodal)', alpha=0.8, color='blue')
        plt.title(f"{ticker}: Trimodal Price Prediction (Acc: {dir_acc:.2%})")
        plt.legend()
        plt.savefig(REMOTE_MODEL_PATH / f"{ticker}_price_reconstruction.png")
        plt.close()

        print(f"generating graphs for {ticker_name}...")
        visualize_results(ticker_name,val_targets,val_predictions,save_dir="experiment_plots")
        
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