import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings from mplfinance about too much data, as we generate small snippets
warnings.filterwarnings("ignore")

def transform_to_gaf(series):
    """
    Transforms a 1D time series into a Gramian Angular Summation Field (GASF) matrix.
    Input: pandas Series or numpy array
    Output: 2D numpy array (Image)
    """
    # 1. Normalize the series to [-1, 1] range (Required for arccos)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Reshape is necessary for scaler (N, 1)
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    
    # 2. Convert to Polar Coordinates
    # Phi (Angle) = arccos(normalized_value)
    phi = np.arccos(scaled_series) 
    
    # 3. Calculate GASF
    # Formula: cos(phi_i + phi_j) = cos(phi_i)cos(phi_j) - sin(phi_i)sin(phi_j)
    # Since scaled_series = cos(phi), we can compute this efficiently using outer products
    gaf_matrix = np.outer(scaled_series, scaled_series) - np.outer(np.sin(phi), np.sin(phi))
    
    return gaf_matrix

def generate_ts_images(df: pd.DataFrame, output_dir: Path, sequence_length=60):
    """
    Iterates through the DataFrame and generates a GAF image for every rolling window.
    
    Args:
        df: DataFrame containing 'Open', 'High', 'Low', 'Close'
        output_dir: Path object where images will be saved
        sequence_length: The lookback period (must match model sequence length)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    close_series = df['Close']
    
    # Check if we have enough data
    if len(df) <= sequence_length:
        print(f"⚠️ Not enough data to generate images. Need > {sequence_length} rows.")
        return

    # Iterate through the data to create rolling windows
    # We start at sequence_length so we have a full window of past data
    for i in range(sequence_length, len(df)):
        
        # The date corresponding to the END of the sequence (the prediction point)
        current_date = df.index[i].strftime('%Y-%m-%d')
        target_file = output_dir / f"{current_date}_gaf.png"
        
        # Skip if already exists (resume capability)
        if target_file.exists():
            continue

        # Get the window of data
        window_series = close_series.iloc[i-sequence_length : i]
        
        # --- Method: Gramian Angular Field (GAF) ---
        # We use GAF because it preserves temporal correlation better than raw plots for CNNs
        gaf_matrix = transform_to_gaf(window_series)
        
        # Plot and save
        plt.figure(figsize=(4, 4)) # Small size is sufficient for 224x224 resize
        plt.imshow(gaf_matrix, cmap='rainbow', origin='lower', aspect='auto')
        plt.axis('off') # Strictly remove axes/ticks/labels
        plt.tight_layout(pad=0)
        
        plt.savefig(target_file, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

    print(f"   ✅ Finished generating images in {output_dir}")