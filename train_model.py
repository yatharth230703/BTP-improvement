import modal
import os
import sys
from pathlib import Path

# --- Configuration ---
APP_NAME = "fintech-research-training"
VOLUME_NAME = "fintech-data-vol"
REMOTE_MOUNT_PATH = Path("/vol")
REMOTE_DATA_PATH = REMOTE_MOUNT_PATH / "processed"
REMOTE_MODEL_PATH = REMOTE_MOUNT_PATH / "models"
LOCAL_DATA_DIR = Path("data/processed")
LOCAL_SRC_DIR = Path("src")

# --- Define the Modal Image ---
# FIX 1: Use NVIDIA Base Image to ensure CuPy/XGBoost compatibility
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "torch",
        "pandas",
        "numpy",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "xgboost",
        "cupy-cuda12x"
    )
    .add_local_python_source(str(LOCAL_SRC_DIR))
)

# --- Define the App & Volume ---
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- Helper: Data Check ---
@app.function(volumes={REMOTE_MOUNT_PATH: volume})
def needs_upload():
    return not (REMOTE_DATA_PATH).exists()

# --- The Remote Training Function ---
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
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import cupy as cp
    import sys
    
    from src.models.dataset import FinancialDataset
    from src.models.lstm import DualBranchLSTM
    from src.models.transformer import TimeSeriesTransformer
    from src.features.financial import create_classification_target, select_features

    print(f"🚀 Starting Grand Unified Training (Weighted + Tuned) on {torch.cuda.get_device_name(0)}")
    
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 0.001
    SEQUENCE_LENGTH = 60
    
    REMOTE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    results = {}

    for ticker in target_stocks:
        print(f"\n{'='*60}")
        print(f"🏢 STARTING ENSEMBLE TRAINING FOR: {ticker}")
        print(f"{'='*60}")
        
        # 1. DATA PREPARATION
        stock_dir = REMOTE_DATA_PATH / ticker
        full_data_path = stock_dir / "full_cleaned.csv"
        
        if not full_data_path.exists():
            print(f"⚠️ Data for {ticker} not found. Skipping.")
            continue
            
        print("🛠️ Transforming Data: Creating Classification Targets...")
        df = pd.read_csv(full_data_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        # Create Triple-Barrier Target
        df = create_classification_target(df)
        
        print("🧠 Running Recursive Feature Elimination (RFE)...")
        df_selected = select_features(df, target_col='Target_Direction', n_features=15)
        
        # Re-Split Data
        train_size = int(len(df_selected) * 0.7)
        val_size = int(len(df_selected) * 0.15)
        
        train_df = df_selected.iloc[:train_size]
        val_df = df_selected.iloc[train_size : train_size + val_size]
        test_df = df_selected.iloc[train_size + val_size :]
        
        # Calculate Class Weights
        n_total = len(train_df)
        n_0 = len(train_df[train_df['Target_Direction'] == 0])
        n_1 = len(train_df[train_df['Target_Direction'] == 1])
        
        if n_1 == 0:
            print("⚠️ Warning: No 'Buy' signals. Skipping.")
            continue

        w0 = n_total / (2 * n_0)
        w1 = n_total / (2 * n_1)
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).cuda()
        xgb_scale_pos_weight = n_0 / n_1
        
        print(f"⚖️ Class Balance: 0s={n_0}, 1s={n_1} | Weights: 0={w0:.2f}, 1={w1:.2f}")
        
        # Save temp CSVs
        temp_train_path = stock_dir / "temp_ensemble_train.csv"
        temp_val_path = stock_dir / "temp_ensemble_val.csv"
        temp_test_path = stock_dir / "temp_ensemble_test.csv"
        
        train_df.to_csv(temp_train_path)
        val_df.to_csv(temp_val_path)
        test_df.to_csv(temp_test_path)
        
        # Create Datasets
        train_dataset = FinancialDataset(str(temp_train_path), sequence_length=SEQUENCE_LENGTH, target_col='Target_Direction')
        val_dataset = FinancialDataset(str(temp_val_path), sequence_length=SEQUENCE_LENGTH, target_col='Target_Direction')
        test_dataset = FinancialDataset(str(temp_test_path), sequence_length=SEQUENCE_LENGTH, target_col='Target_Direction')
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        input_dim = train_dataset.get_input_dim()

        # --- 2. TRAIN EXPERT A: LSTM (Weighted) ---
        print("\n🥋 Training Expert A: Dual-Branch LSTM...")
        model_lstm = DualBranchLSTM(input_dim=input_dim, output_dim=2).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model_lstm.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            model_lstm.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.cuda(), y_b.long().cuda()
                optimizer.zero_grad()
                pred = model_lstm(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
        
        torch.save(model_lstm.state_dict(), REMOTE_MODEL_PATH / f"{ticker}_lstm.pth")
        
        # --- 3. TRAIN EXPERT B: TRANSFORMER (Weighted) ---
        print("🤖 Training Expert B: Time-Series Transformer...")
        model_transformer = TimeSeriesTransformer(input_dim=input_dim, output_dim=2).cuda()
        optimizer_t = optim.Adam(model_transformer.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            model_transformer.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.cuda(), y_b.long().cuda()
                optimizer_t.zero_grad()
                pred = model_transformer(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer_t.step()
                
        torch.save(model_transformer.state_dict(), REMOTE_MODEL_PATH / f"{ticker}_transformer.pth")

        # --- 4. TRAIN EXPERT C: XGBOOST (Weighted) ---
        print("🌲 Training Expert C: XGBoost (GPU Accelerated)...")
        
        def get_numpy_data(loader):
            X_list, y_list = [], []
            for X_b, y_b in loader:
                flat_X = X_b.view(X_b.size(0), -1).numpy()
                X_list.append(flat_X)
                y_list.append(y_b.numpy())
            if not X_list: return np.array([]), np.array([])
            return np.concatenate(X_list), np.concatenate(y_list)

        X_train_np, y_train_np = get_numpy_data(train_loader)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            device="cuda",
            scale_pos_weight=xgb_scale_pos_weight
        )
        xgb_model.fit(X_train_np, y_train_np)
        
        # --- 5. META-LEARNER & THRESHOLD TUNING ---
        print("🧠 Training Meta-Learner & Tuning Threshold...")
        
        model_lstm.eval()
        model_transformer.eval()
        
        # FIX 2: Initialize lists outside the loop to prevent UnboundLocalError
        meta_X_val = []
        meta_y_val = []
        
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_cuda = X_b.cuda()
                
                p_lstm = torch.softmax(model_lstm(X_cuda), dim=1)[:, 1].cpu().numpy()
                p_trans = torch.softmax(model_transformer(X_cuda), dim=1)[:, 1].cpu().numpy()
                
                flat_X_np = X_b.view(X_b.size(0), -1).numpy()
                flat_X_cp = cp.asarray(flat_X_np)
                p_xgb = xgb_model.predict_proba(flat_X_cp)[:, 1]
                
                stacked = np.column_stack((p_lstm, p_trans, p_xgb))
                meta_X_val.append(stacked)
                meta_y_val.append(y_b.numpy())
        
        if len(meta_X_val) == 0:
            print("⚠️ Validation Set empty. Skipping.")
            continue
            
        meta_X_val = np.concatenate(meta_X_val)
        meta_y_val = np.concatenate(meta_y_val)
        
        # Train Meta-Learner
        meta_learner = LogisticRegression(class_weight='balanced')
        meta_learner.fit(meta_X_val, meta_y_val)
        
        # Find Optimal Threshold (Maximizing G-Mean)
        val_probs = meta_learner.predict_proba(meta_X_val)[:, 1]
        best_threshold = 0.5
        best_gmean = 0.0
        
        for thresh in np.arange(0.3, 0.7, 0.01):
            preds = (val_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(meta_y_val, preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmean = np.sqrt(sensitivity * specificity)
            if gmean > best_gmean:
                best_gmean = gmean
                best_threshold = thresh
        
        print(f"   ⚖️ Optimal Threshold: {best_threshold:.2f} (G-Mean: {best_gmean:.4f})")
        
        # --- 6. FINAL EVALUATION ---
        print("📊 Final Evaluation on Test Set...")
        
        # FIX 2: Initialize lists outside the loop
        meta_X_test = []
        meta_y_test = []
        
        with torch.no_grad():
            for i, (X_b, y_b) in enumerate(test_loader):
                X_cuda = X_b.cuda()
                p_lstm = torch.softmax(model_lstm(X_cuda), dim=1)[:, 1].cpu().numpy()
                p_trans = torch.softmax(model_transformer(X_cuda), dim=1)[:, 1].cpu().numpy()
                
                flat_X_np = X_b.view(X_b.size(0), -1).numpy()
                flat_X_cp = cp.asarray(flat_X_np)
                p_xgb = xgb_model.predict_proba(flat_X_cp)[:, 1]
                
                stacked = np.column_stack((p_lstm, p_trans, p_xgb))
                meta_X_test.append(stacked)
                meta_y_test.append(y_b.numpy())

        if len(meta_X_test) == 0:
            print("⚠️ Test Set empty. Skipping.")
            continue

        meta_X_test = np.concatenate(meta_X_test)
        meta_y_test = np.concatenate(meta_y_test)
        
        # 1. Get Raw Probabilities
        raw_test_probs = meta_learner.predict_proba(meta_X_test)[:, 1]
        
        # 2. FIX 3: Apply Signal Smoothing (EMA)
        smoothed_probs = pd.Series(raw_test_probs).ewm(span=3).mean().values
        
        # 3. Apply Optimal Threshold
        final_preds = (smoothed_probs >= best_threshold).astype(int)
        
        acc = accuracy_score(meta_y_test, final_preds)
        f1 = f1_score(meta_y_test, final_preds)
        r2 = r2_score(meta_y_test, smoothed_probs)
        
        cm = confusion_matrix(meta_y_test, final_preds)
        print(f"   Confusion Matrix: \n{cm}")
        
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(meta_y_test, label='Actual (1=Up)', marker='.', linestyle='', alpha=0.3, color='black')
        plt.plot(raw_test_probs, label='Raw Prob', color='lightblue', alpha=0.5, linewidth=1)
        plt.plot(smoothed_probs, label='Smoothed Prob (EMA)', color='blue', linewidth=2)
        plt.axhline(y=best_threshold, color='r', linestyle='--', label=f'Threshold ({best_threshold:.2f})')
        plt.title(f"{ticker}: Ensemble Prediction (Weighted & Smoothed)")
        plt.legend()
        
        plot_path = REMOTE_MODEL_PATH / f"{ticker}_ensemble_prediction.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✅ {ticker} FINAL RESULTS:")
        print(f"   Ensemble Accuracy: {acc:.2%}")
        print(f"   Ensemble F1 Score: {f1:.4f}")
        print(f"   Ensemble R2 Score: {r2:.4f}")
        
        results[ticker] = {"Accuracy": acc, "F1": f1, "R2": r2}

    print("\n🏆 GRAND UNIFIED LEADERBOARD")
    print(f"{'Ticker':<12} | {'Accuracy':<10} | {'F1 Score':<10} | {'R2 Score':<10}")
    print("-" * 55)
    for t, m in results.items():
        print(f"{t:<12} | {m['Accuracy']:.2%}     | {m['F1']:.4f}     | {m['R2']:.4f}")
        
    return results

# --- Local Entrypoint ---
@app.local_entrypoint()
def main():
    print("🔄 Verifying Data...")
    should_upload = needs_upload.remote()

    if should_upload:
        print("📤 Uploading data (if needed)...")
        vol = modal.Volume.from_name(VOLUME_NAME)
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(LOCAL_DATA_DIR, remote_path="processed", recursive=True)
    
    TARGET_STOCKS = ['SBIN', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'PNB']
    train_remote.remote(TARGET_STOCKS)