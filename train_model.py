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

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(volumes={REMOTE_MOUNT_PATH: volume})
def needs_upload():
    return not (REMOTE_DATA_PATH).exists()

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
    from sklearn.metrics import r2_score, mean_squared_error 
    from sklearn.linear_model import LinearRegression 
    from sklearn.preprocessing import RobustScaler # FIX 1: RobustScaler
    import xgboost as xgb
    import cupy as cp
    import sys
    from sklearn.preprocessing import MinMaxScaler
    from src.models.dataset import FinancialDataset
    from src.models.lstm import DualBranchLSTM
    from src.models.transformer import TimeSeriesTransformer
    from src.features.financial import create_regression_target as create_target, select_features

    print(f"🚀 Starting Grand Unified Training (ROBUST REGRESSION) on {torch.cuda.get_device_name(0)}")
    
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
            
        print("🛠️ Transforming Data: Creating Regression Targets...")
        df = pd.read_csv(full_data_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        df = create_target(df) 
        
        print("⚠️ DIAGNOSTIC MODE: RFE Disabled. Using ALL engineered features.")
        df_selected = df
        
        # Re-Split Data
        train_size = int(len(df_selected) * 0.7)
        val_size = int(len(df_selected) * 0.15)
        
        train_df = df_selected.iloc[:train_size].copy()
        val_df = df_selected.iloc[train_size : train_size + val_size].copy()
        test_df = df_selected.iloc[train_size + val_size :].copy()
        
        from sklearn.preprocessing import MinMaxScaler

        TARGET_COL = 'Target_Close'
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        train_df[TARGET_COL] = target_scaler.fit_transform(train_df[[TARGET_COL]])
        val_df[TARGET_COL] = target_scaler.transform(val_df[[TARGET_COL]])
        test_df[TARGET_COL] = target_scaler.transform(test_df[[TARGET_COL]])
        
        print(f"   ⚖️ Target '{TARGET_COL}' Scaled (MinMax). Range: 0-1")        
        # Save temp CSVs
        temp_train_path = stock_dir / "temp_ensemble_train.csv"
        temp_val_path = stock_dir / "temp_ensemble_val.csv"
        temp_test_path = stock_dir / "temp_ensemble_test.csv"
        
        

        train_df.to_csv(temp_train_path)
        val_df.to_csv(temp_val_path)
        test_df.to_csv(temp_test_path)
        
        train_dataset = FinancialDataset(str(temp_train_path), sequence_length=SEQUENCE_LENGTH, target_col=TARGET_COL)
        val_dataset = FinancialDataset(str(temp_val_path), sequence_length=SEQUENCE_LENGTH, target_col=TARGET_COL)
        test_dataset = FinancialDataset(str(temp_test_path), sequence_length=SEQUENCE_LENGTH, target_col=TARGET_COL)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        input_dim = train_dataset.get_input_dim()

        # --- 2. TRAIN EXPERT A: LSTM ---
        print("\n🥋 Training Expert A: Dual-Branch LSTM (Regression)...")
        model_lstm = DualBranchLSTM(input_dim=input_dim, output_dim=1).cuda()
        
        # FIX 2: HuberLoss handles outliers better than MSE
        criterion = nn.HuberLoss(delta=1.0) 
        optimizer = optim.Adam(model_lstm.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            model_lstm.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.cuda(), y_b.float().cuda().unsqueeze(1)
                optimizer.zero_grad()
                pred = model_lstm(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
        
        torch.save(model_lstm.state_dict(), REMOTE_MODEL_PATH / f"{ticker}_lstm.pth")
        
        # --- 3. TRAIN EXPERT B: TRANSFORMER ---
        print("🤖 Training Expert B: Time-Series Transformer (Regression)...")
        model_transformer = TimeSeriesTransformer(input_dim=input_dim, output_dim=1).cuda()
        optimizer_t = optim.Adam(model_transformer.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            model_transformer.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.cuda(), y_b.float().cuda().unsqueeze(1)
                optimizer_t.zero_grad()
                pred = model_transformer(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer_t.step()
                
        torch.save(model_transformer.state_dict(), REMOTE_MODEL_PATH / f"{ticker}_transformer.pth")

        # --- 4. TRAIN EXPERT C: XGBOOST ---
        print("🌲 Training Expert C: XGBoost (GPU Accelerated, Regression)...")
        
        def get_numpy_data(loader):
            X_list, y_list = [], []
            for X_b, y_b in loader:
                flat_X = X_b.view(X_b.size(0), -1).numpy()
                X_list.append(flat_X)
                y_list.append(y_b.numpy())
            if not X_list: return np.array([]), np.array([])
            return np.concatenate(X_list), np.concatenate(y_list)

        X_train_np, y_train_np = get_numpy_data(train_loader)
        
        xgb_model = xgb.XGBRegressor( 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            device="cuda",
        )
        xgb_model.fit(X_train_np, y_train_np)

        # ... (Feature Importance Plotting logic remains the same) ...
        print("🔍 Generating Feature Importance Plot...")
        try:
            feature_names = [c for c in df_selected.columns if c not in [TARGET_COL, 'Date', 'Unnamed: 0']]
            num_base_features = len(feature_names)
            importance_map = xgb_model.get_booster().get_score(importance_type='gain')
            agg_importance = {}
            for feat_key, score in importance_map.items():
                idx = int(feat_key.replace('f', '')) 
                base_idx = idx % num_base_features
                if base_idx < len(feature_names):
                    name = feature_names[base_idx]
                    agg_importance[name] = agg_importance.get(name, 0) + score

            if agg_importance:
                sorted_feats = sorted(agg_importance.items(), key=lambda x: x[1], reverse=True)
                keys = [x[0] for x in sorted_feats]
                values = [x[1] for x in sorted_feats]
                
                plt.figure(figsize=(10, 8))
                plt.barh(keys[:20], values[:20])
                plt.gca().invert_yaxis()
                plt.title(f"Top Features Driving {ticker} (Aggregated Gain)")
                plt.tight_layout()
                plot_path = REMOTE_MODEL_PATH / f"{ticker}_feature_importance.png"
                plt.savefig(plot_path)
                plt.close()
                print(f"   📊 Saved Importance Plot to {plot_path}")
        except Exception as e:
            print(f"   ⚠️ Could not plot feature importance: {e}")
        
        # --- 5. META-LEARNER ---
        print("🧠 Training Meta-Learner (Linear Regression)...")
        
        model_lstm.eval()
        model_transformer.eval()
        
        meta_X_val = []
        meta_y_val = []
        
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_cuda = X_b.cuda()
                
                p_lstm = model_lstm(X_cuda).squeeze().cpu().numpy()
                p_trans = model_transformer(X_cuda).squeeze().cpu().numpy()
                
                flat_X_np = X_b.view(X_b.size(0), -1).numpy()
                flat_X_cp = cp.asarray(flat_X_np)
                p_xgb = xgb_model.predict(flat_X_cp) 
                
                stacked = np.column_stack((p_lstm, p_trans, p_xgb))
                meta_X_val.append(stacked)
                meta_y_val.append(y_b.numpy())
        
        if len(meta_X_val) == 0:
            print("⚠️ Validation Set empty. Skipping.")
            continue
            
        meta_X_val = np.concatenate(meta_X_val)
        meta_y_val = np.concatenate(meta_y_val)
        
        meta_learner = LinearRegression()
        meta_learner.fit(meta_X_val, meta_y_val)
        
        # --- 6. FINAL EVALUATION ---
        print("📊 Final Evaluation on Test Set...")
        
        meta_X_test = []
        meta_y_test = []
        
        with torch.no_grad():
            for i, (X_b, y_b) in enumerate(test_loader):
                X_cuda = X_b.cuda()
                p_lstm = model_lstm(X_cuda).squeeze().cpu().numpy()
                p_trans = model_transformer(X_cuda).squeeze().cpu().numpy()
                
                flat_X_np = X_b.view(X_b.size(0), -1).numpy()
                flat_X_cp = cp.asarray(flat_X_np)
                p_xgb = xgb_model.predict(flat_X_cp) 
                
                stacked = np.column_stack((p_lstm, p_trans, p_xgb))
                meta_X_test.append(stacked)
                meta_y_test.append(y_b.numpy())

        if len(meta_X_test) == 0:
            print("⚠️ Test Set empty. Skipping.")
            continue

        meta_X_test = np.concatenate(meta_X_test)
        meta_y_test = np.concatenate(meta_y_test)
        
        scaled_predictions = meta_learner.predict(meta_X_test)
        
        # Inverse Transform
        final_predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
        actual_values = target_scaler.inverse_transform(meta_y_test.reshape(-1, 1)).flatten()
        
        r2 = r2_score(actual_values, final_predictions)
        mse = mean_squared_error(actual_values, final_predictions)
        rmse = np.sqrt(mse)
        
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(actual_values, label='Actual Log Return', alpha=0.6, color='black')
        plt.plot(final_predictions, label='Predicted Log Return', alpha=0.8, color='blue', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Return')
        plt.title(f"{ticker}: Ensemble Regression (Robust Scaler)")
        plt.legend()
        
        plot_path = REMOTE_MODEL_PATH / f"{ticker}_regression_prediction.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✅ {ticker} FINAL RESULTS:")
        print(f"   Ensemble R2 Score: {r2:.4f}")
        print(f"   Ensemble RMSE: {rmse:.6f}")
        
        results[ticker] = {"R2": r2, "RMSE": rmse}

    print("\n🏆 GRAND UNIFIED REGRESSION LEADERBOARD")
    print(f"{'Ticker':<12} | {'R2 Score':<10} | {'RMSE':<10}")
    print("-" * 35)
    for t, m in results.items():
        print(f"{t:<12} | {m['R2']:.4f}     | {m['RMSE']:.6f}")
        
    return results

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