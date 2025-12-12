import pandas as pd
import os
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_STOCKS, TRAIN_START, TEST_END
from src.features.financial import (
    calculate_technical_indicators, 
    scale_data, 
    create_classification_target,
    calculate_rolling_beta
)
from src.features.corporate import apply_corporate_actions
from src.features.news import process_sentiment
from src.utils.visualization import plot_price_vs_sentiment, plot_correlation_heatmap

# NOTE: Removed 'from sklearn.decomposition import PCA' as it is no longer needed.

# --- Helper Function: Process Explicit FinBERT Scores ---
def process_sentiment_scores():
    """
    Loads the explicit FinBERT scores (Sentiment_Score, Prob_Pos, Prob_Neg, News_Volume).
    """
    print("   📊 Loading Daily FinBERT Scores...")
    score_path = os.path.join(PROCESSED_DATA_DIR, 'daily_finbert_scores.csv')
    
    if not os.path.exists(score_path):
        print("⚠️ Scores not found locally. Run 'modal volume get...' first.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(score_path, index_col='Date', parse_dates=True)
        # Select the columns we need for the model
        return df[['Sentiment_Score', 'Prob_Pos', 'Prob_Neg', 'News_Volume']]
    except Exception as e:
        print(f"⚠️ Error reading scores file: {e}")
        return pd.DataFrame()

# --- Helper Function: Process Sector Index (Nifty Bank) ---
# (This remains unchanged as it is a working and necessary feature)
def process_sector_index():
    """
    Loads and processes the Nifty Bank Index data.
    """
    print("   📊 Processing Nifty Bank Index (Sector Context)...")
    try:
        idx_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Nifty Bank Historical Data.csv'))
        idx_df['Date'] = pd.to_datetime(idx_df['Date'])
        idx_df.set_index('Date', inplace=True)
        
        if idx_df.index.tz is not None:
            idx_df.index = idx_df.index.tz_localize(None)
            
        for col in ['Price', 'Open', 'High', 'Low']:
            if col in idx_df.columns and idx_df[col].dtype == object:
                idx_df[col] = idx_df[col].str.replace(',', '', regex=False).astype(float)
        
        if 'Price' in idx_df.columns:
            idx_df.rename(columns={'Price': 'Close'}, inplace=True)
            
        idx_df['Sector_Ret'] = idx_df['Close'].pct_change()
        idx_df['Sector_Vol'] = idx_df['Sector_Ret'].rolling(window=20).std()
        
        return idx_df[['Sector_Ret', 'Sector_Vol']]
        
    except Exception as e:
        print(f"⚠️ Warning: Could not process Sector Index. Error: {e}")
        return None

# --- Main Pipeline Execution ---
def main():
    print("Starting Data Pipeline...")
    
    # FIX 1: Initialize variables outside try/except for proper scope (NameError fix)
    news_df = None
    corp_df = None
    
    # 1. Load Common Data
    print("Loading Auxiliary Data...")
    try:
        news_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Enriched_IndianFinancialNews.csv'))
        news_df['Date'] = pd.to_datetime(news_df['Date'])
        corp_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'corporate_actions_2005_2020.csv'))
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure CSVs are in data/raw/")
        return

    # Check if news_df was successfully loaded
    if news_df is None:
        print("Error: News data failed to load.")
        return
        
    sentiment_series = process_sentiment(news_df, TRAIN_START, TEST_END)
    sector_df = process_sector_index()
    
    # --- Load Explicit FinBERT Scores ---
    sentiment_score_df = process_sentiment_scores()
    # ------------------------------------

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
        
        # B. Financial Feature Engineering (RSI, MACD, Log_Ret)
        df = calculate_technical_indicators(df)
        
        # C. Merge Traditional Sentiment
        df = df.join(sentiment_series['Sentiment_Decay'], how='left')
        df['Sentiment_Decay'].fillna(0, inplace=True)
        
        # --- Merge Explicit FinBERT Scores ---
        if not sentiment_score_df.empty:
            df = df.join(sentiment_score_df, how='left')
            # Fill days with no news with 0 (Neutral score, 0 volume)
            df[['Sentiment_Score', 'Prob_Pos', 'Prob_Neg', 'News_Volume']] = \
                df[['Sentiment_Score', 'Prob_Pos', 'Prob_Neg', 'News_Volume']].fillna(0.0)
            print("   Merged Explicit FinBERT Scores.")
        # --------------------------------------
        
        # --- Merge Sector Context and Calculate Beta ---
        if sector_df is not None: 
            df = df.join(sector_df, how='left') 
            df[['Sector_Ret', 'Sector_Vol']] = df[['Sector_Ret', 'Sector_Vol']].ffill()
            
            # Calculate the explicit Beta factor 
            df = calculate_rolling_beta(df, window=60)
            print(f"   Beta Calculated.")
        # -----------------------------------------------
        
        # D. Target Creation (Triple-Barrier)
        df = create_classification_target(df, barrier_multiplier=0.5)
        
        # E. Scaling
        features_to_scale = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Log_Ret', 'RSI', 'MACD', 
            'MACD_Signal', 'Volatility', 'Sentiment_Decay', 'Dividend_Yield',
            'Sector_Ret', 'Sector_Vol', 'Rolling_Beta',
            # Add new explicit FinBERT scores to the scaling list
            'Sentiment_Score', 'Prob_Pos', 'Prob_Neg', 'News_Volume'
        ]

        # Final cleanup before scaling
        df.dropna(inplace=True) 

        df, scaler = scale_data(df, features_to_scale)
        print(f"   ⚖️ Scaled data for {ticker}. Target direction created.")
        
        # F. Visualization
        plot_price_vs_sentiment(df.copy(), ticker)
        plot_correlation_heatmap(df.copy(), ticker)
        
        # G. Save the FINAL, FULL SCALED & CLASSIFIED Data
        stock_dir = os.path.join(PROCESSED_DATA_DIR, ticker)
        os.makedirs(stock_dir, exist_ok=True)

        df.to_csv(os.path.join(stock_dir, 'full_cleaned.csv'))
        print(f"✅ Saved FULL SCALED data to {stock_dir}/full_cleaned.csv")
        
    print("Pipeline Complete. Local data is now ready for upload.")

if __name__ == "__main__":
    main()