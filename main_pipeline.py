import pandas as pd
import os
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_STOCKS, TRAIN_START, TEST_END
from src.features.financial import (
    calculate_technical_indicators, 
    scale_data, 
    create_classification_target
)
from src.features.corporate import apply_corporate_actions
from src.features.news import process_sentiment
from src.utils.visualization import plot_price_vs_sentiment, plot_correlation_heatmap

def process_sector_index():
    """
    Loads and processes the Nifty Bank Index data to serve as a global signal.
    """
    print("   📊 Processing Nifty Bank Index (Sector Context)...")
    try:
        # Load Index Data
        idx_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Nifty Bank Historical Data.csv'))
        idx_df['Date'] = pd.to_datetime(idx_df['Date'])
        idx_df.set_index('Date', inplace=True)
        
        # Handle timezone if present
        if idx_df.index.tz is not None:
            idx_df.index = idx_df.index.tz_localize(None)
            
        # Clean numeric columns (sometimes they have commas like "25,000.00")
        for col in ['Price', 'Open', 'High', 'Low']:
            if idx_df[col].dtype == object:
                idx_df[col] = idx_df[col].str.replace(',', '').astype(float)
        
        # Rename 'Price' to 'Close' for consistency if needed
        if 'Price' in idx_df.columns:
            idx_df.rename(columns={'Price': 'Close'}, inplace=True)
            
        # Calculate Sector Signals
        # We perform simple feature engineering for the sector
        idx_df['Sector_Ret'] = idx_df['Close'].pct_change()
        idx_df['Sector_Vol'] = idx_df['Sector_Ret'].rolling(window=20).std()
        idx_df['Sector_RSI'] = calculate_technical_indicators(idx_df)['RSI'] # Re-use our func
        
        # Select only the features we want to inject into other stocks
        return idx_df[['Sector_Ret', 'Sector_Vol', 'Sector_RSI']]
        
    except Exception as e:
        print(f"⚠️ Warning: Could not process Sector Index. Error: {e}")
        return None

def main():
    print("Starting Data Pipeline...")
    
    # 1. Load Common Data
    print("Loading Auxiliary Data...")
    try:
        news_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Enriched_IndianFinancialNews.csv'))
        news_df['Date'] = pd.to_datetime(news_df['Date'])
        corp_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'corporate_actions_2005_2020.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure CSVs are in data/raw/")
        return

    sentiment_series = process_sentiment(news_df, TRAIN_START, TEST_END)
    
    # --- NEW: Process Sector Data ---
    sector_df = process_sector_index()
    # --------------------------------

    # 2. Process Each Stock
    for ticker in TARGET_STOCKS:
        print(f"Processing {ticker}...")
        
        file_path = os.path.join(RAW_DATA_DIR, f'{ticker}_NS_prices_2005_2020.csv')
        if not os.path.exists(file_path):
            print(f"Skipping {ticker}, file not found.")
            continue
            
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        mask = (df.index >= TRAIN_START) & (df.index <= TEST_END)
        df = df.loc[mask].copy()
        
        # A. Apply Corporate Actions
        df = apply_corporate_actions(df, corp_df, ticker)
        
        # B. Financial Feature Engineering
        df = calculate_technical_indicators(df)
        
        # C. Merge Sentiment
        df = df.join(sentiment_series['Sentiment_Decay'], how='left')
        df['Sentiment_Decay'].fillna(0, inplace=True)
        
        # --- NEW: Merge Sector Context ---
        if sector_df is not None:
            df = df.join(sector_df, how='left')
            # Forward fill sector data for any missing days (holidays)
            df[['Sector_Ret', 'Sector_Vol', 'Sector_RSI']] = df[['Sector_Ret', 'Sector_Vol', 'Sector_RSI']].ffill()
            df.dropna(inplace=True) # Drop initial rows where sector data might be NaN due to windows
        # ---------------------------------
        
        # D. Target Creation (Triple-Barrier)
        df = create_classification_target(df, barrier_multiplier=0.5)
        
        # E. Scaling
        # Add new sector features to scaling list
        features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 
                             'Log_Ret', 'RSI', 'MACD', 'MACD_Signal', 
                             'Volatility', 'Sentiment_Decay', 'Dividend_Yield',
                             'Sector_Ret', 'Sector_Vol', 'Sector_RSI'] # <-- Added here
        
        df, scaler = scale_data(df, features_to_scale)
        print(f"   ⚖️ Scaled data for {ticker}. Added Sector Context.")
        
        # F. Visualization
        plot_price_vs_sentiment(df.copy(), ticker)
        plot_correlation_heatmap(df.copy(), ticker)
        
        # G. Save
        stock_dir = os.path.join(PROCESSED_DATA_DIR, ticker)
        os.makedirs(stock_dir, exist_ok=True)
        df.to_csv(os.path.join(stock_dir, 'full_cleaned.csv'))
        print(f"✅ Saved FULL SCALED data to {stock_dir}/full_cleaned.csv")
        
    print("Pipeline Complete. Local data ready for upload.")

if __name__ == "__main__":
    main()