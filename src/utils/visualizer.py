import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

class Visualizer:
    """
    Handles logging of training metrics and generation of evaluation plots.
    Specifically designed to track the Gating Weights of the Trimodal Network.
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.logs = []

    def log_step(self, epoch, train_loss, val_loss, r2_score, gate_weights):
        """
        Records metrics for a single epoch.
        params:
            gate_weights: numpy array of shape (Batch_Size, 3) from the last validation batch.
        """
        # Average the gate weights across the batch to get a "global" sentiment for this epoch
        avg_gates = np.mean(gate_weights, axis=0) 
        
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "r2_score": r2_score,
            "gate_num": avg_gates[0],
            "gate_text": avg_gates[1],
            "gate_img": avg_gates[2]
        }
        self.logs.append(entry)

    def save_reports(self, ticker):
        """
        Generates and saves:
        1. CSV Log file
        2. Training Loss Curve
        3. Modality Contribution Stackplot (The 'Killer Feature')
        """
        if not self.logs:
            print(f"⚠️ No logs to save for {ticker}")
            return

        df = pd.DataFrame(self.logs)
        
        # 1. Save CSV
        csv_path = os.path.join(self.save_dir, f"{ticker}_training_logs.csv")
        df.to_csv(csv_path, index=False)
        
        # 2. Plot Training Curves
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', linestyle='--')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.title(f"{ticker}: Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f"{ticker}_loss_curve.png"))
        plt.close()
        
        # 3. Plot Modality Gates (Stacked Area Chart)
        plt.figure(figsize=(10, 6))
        
        # Use a stackplot to show relative importance summing to 1.0
        plt.stackplot(df['epoch'], 
                      df['gate_num'], df['gate_text'], df['gate_img'], 
                      labels=['Numerical (LSTM)', 'Text (Placeholder)', 'Visual (CNN)'],
                      colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                      alpha=0.8)
        
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f"{ticker}: Modality Importance (Gating Weights) Over Time")
        plt.ylabel("Importance Weight (Sum=1.0)")
        plt.xlabel("Epoch")
        plt.margins(0, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{ticker}_modality_gates.png"))
        plt.close()