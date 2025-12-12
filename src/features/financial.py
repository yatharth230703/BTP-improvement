import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def calculate_technical_indicators(df):
    """
    Adds RSI, MACD, and Log Returns to the dataframe.
    Expects columns: ['Close', 'Volume']
    """
    df = df.copy()
    
    # 1. Log Returns (Stationarity)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. RSI (Relative Strength Index) - 14 day
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Volatility (20-day Rolling Std Dev)
    df['Volatility'] = df['Log_Ret'].rolling(window=20).std()
    
    # Drop NaN created by rolling windows
    df.dropna(inplace=True)
    
    return df
def scale_data(df, feature_columns):
    """
    Scales numerical features to [0, 1] range to ensure LSTM convergence.
    """
    scaler = MinMaxScaler()
    
    # We only scale the columns that exist in the dataframe
    cols_to_scale = [c for c in feature_columns if c in df.columns]
    
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df, scaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
def create_classification_target(df, window=20, barrier_multiplier=0.5):
    """
    Implements a simplified Triple-Barrier Method.
    Target = 1 only if the next day's return exceeds a volatility-based threshold.
    Target = 0 for small moves (noise) or negative moves.
    
    Args:
        df: Dataframe with 'Close' column.
        window: Lookback window for volatility calculation.
        barrier_multiplier: How many standard deviations constitutes a 'signal'.
    """
    df = df.copy()
    
    # 1. Calculate Daily Returns
    df['Returns'] = df['Close'].pct_change()
    
    # 2. Calculate Dynamic Volatility (Rolling Standard Deviation)
    # This tells us how much the stock typically moves in a day.
    df['Volatility'] = df['Returns'].rolling(window=window).std()
    
    # 3. Define the Barrier (Threshold)
    # If Volatility is 2%, and multiplier is 0.5, we need a >1% move to call it a "Buy".
    df['Barrier'] = df['Volatility'] * barrier_multiplier
    
    # 4. Create the Target
    # We shift(-1) because we want to predict tomorrow's move using today's data.
    # Logic: Is Tomorrow's Return > Tomorrow's Barrier?
    future_return = df['Returns'].shift(-1)
    
    # Target = 1 (Significant Up Move), 0 (Noise or Down)
    df['Target_Direction'] = (future_return > df['Barrier']).astype(int)
    
    # Drop NaNs created by rolling window and shift
    df.dropna(inplace=True)
    
    # Clean up temporary columns
    df.drop(columns=['Returns', 'Barrier'], inplace=True)
    
    return df

def select_features(df, target_col='Target_Direction', n_features=10):
    """
    Uses Recursive Feature Elimination to pick the best features.
    """
    # Simple linear model for selection
    model = LogisticRegression(solver='liblinear')
    
    # Drop non-numeric and target
    X = df.drop(columns=[target_col, 'Date'], errors='ignore').select_dtypes(include=['number'])
    y = df[target_col]
    
    selector = RFE(model, n_features_to_select=n_features)
    selector = selector.fit(X, y)
    
    selected_cols = X.columns[selector.support_].tolist()
    print(f"✨ Selected Best {n_features} Features: {selected_cols}")
    
    # Return dataframe with only selected features + Target
    return df[selected_cols + [target_col]]

def calculate_rolling_beta(df, window=60):
    """
    Calculates the Rolling Beta of the stock relative to the Sector Index.
    Beta = Covariance(Stock_Ret, Sector_Ret) / Variance(Sector_Ret)
    """
    # Ensure Sector_Ret exists (it was merged in main_pipeline)
    if 'Sector_Ret' not in df.columns:
        return df
        
    # 1. Calculate Rolling Covariance
    # We use a 60-day window (approx 3 trading months) to capture medium-term sensitivity
    rolling_cov = df['Log_Ret'].rolling(window=window).cov(df['Sector_Ret'])
    
    # 2. Calculate Rolling Variance of the Sector
    rolling_var = df['Sector_Ret'].rolling(window=window).var()
    
    # 3. Compute Beta
    df['Rolling_Beta'] = rolling_cov / rolling_var
    
    # Forward fill initialization NaNs to prevent data loss
    df['Rolling_Beta'] = df['Rolling_Beta'].ffill().fillna(1.0) # Default to 1.0 (market performer)
    
    return df
def create_regression_target(df):
    """
    Creates a Regression Target: The Closing Price of the next day.
    Matches Karadas et al. (2025) methodology for high R2.
    """
    df = df.copy()
    
    # Target: The actual Closing Price of the NEXT day (t+1)
    df['Target_Close'] = df['Close'].shift(-1)
    
    # Drop the final row (future is unknown)
    df.dropna(inplace=True)
    
    return df