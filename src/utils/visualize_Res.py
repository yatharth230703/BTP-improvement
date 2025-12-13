import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def visualize_results(ticker, true_prices, pred_prices, save_dir='plots'):
    """
    Generates a 3-panel dashboard to diagnose the Directional Accuracy Paradox.
    
    Args:
        ticker (str): Name of the stock (e.g., 'SBIN')
        true_prices (np.array): Ground truth prices
        pred_prices (np.array): Model predicted prices
        save_dir (str): Directory to save the plot
    """
    
    # Ensure inputs are flat numpy arrays
    if isinstance(true_prices, torch.Tensor):
        true_prices = true_prices.detach().cpu().numpy().flatten()
    if isinstance(pred_prices, torch.Tensor):
        pred_prices = pred_prices.detach().cpu().numpy().flatten()
        
    # Calculate Returns (percentage change)
    # We use [1:] because we need a previous day to calc return
    true_returns = np.diff(true_prices) / true_prices[:-1]
    pred_returns = np.diff(pred_prices) / pred_prices[:-1]
    
    # Calculate Directions (1 for Up, 0 for Down)
    true_dir = (true_returns > 0).astype(int)
    pred_dir = (pred_returns > 0).astype(int)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'Trimodal Model Diagnostics: {ticker}', fontsize=16, fontweight='bold')

    # 1. Price Comparison (Zoomed into last 100 points for clarity)
    # This reveals the "Lag" - if Red line is just Blue line shifted right, model is chasing.
    ax1 = fig.add_subplot(2, 2, 1)
    zoom = 100 # Last 100 days
    ax1.plot(true_prices[-zoom:], label='Actual Price', color='blue', linewidth=2, alpha=0.7)
    ax1.plot(pred_prices[-zoom:], label='Predicted Price', color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.set_title(f'Price Action Lag Check (Last {zoom} Days)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Directional Confusion Matrix
    # Shows: Are we biased towards buying (False Positives) or Selling?
    ax2 = fig.add_subplot(2, 2, 2)
    cm = confusion_matrix(true_dir, pred_dir)
    # Normalize by row (True labels) to see recall per class
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'Directional Accuracy Confusion Matrix\n(Total Acc: {np.mean(true_dir == pred_dir):.2%})')

    # 3. Returns Scatter Plot (The Quadrant Check)
    # Ideally, points should cluster in Top-Right and Bottom-Left.
    # If they cluster on the x-axis, model is predicting ~0 return.
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.scatter(true_returns, pred_returns, alpha=0.5, color='purple')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.axvline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Actual Returns')
    ax3.set_ylabel('Predicted Returns')
    ax3.set_title('Returns Correlation (Target: Top-Right & Bottom-Left Quadrants)')
    ax3.grid(True, alpha=0.3)

    # Add Quadrant Labels
    ax3.text(max(true_returns)*0.7, max(pred_returns)*0.7, 'Correct UP', color='green', fontweight='bold')
    ax3.text(min(true_returns)*0.7, min(pred_returns)*0.7, 'Correct DOWN', color='green', fontweight='bold')
    ax3.text(min(true_returns)*0.7, max(pred_returns)*0.7, 'False UP', color='red', fontweight='bold')
    ax3.text(max(true_returns)*0.7, min(pred_returns)*0.7, 'False DOWN', color='red', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    import os
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{ticker}_diagnostics.png"
    plt.savefig(save_path)
    print(f"📊 Visualization saved to: {save_path}")
    plt.show()
