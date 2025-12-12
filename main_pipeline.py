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
from sklearn.decomposition import PCA

# --- Helper Function: Process Semantic Embeddings (FinBERT PCA) ---
def process_semantic_embeddings(num_components=10):
    """
    Loads ProsusAI embeddings (simulated read from local storage)
    and reduces dimensionality using PCA.
    """
    print(f"🧠 Loading FinBERT Embeddings for PCA (Reducing to {num_components} components)...")
    
    embedding_path = os.path.join(PROCESSED_DATA_DIR, 'daily_embeddings_prosus.csv')
    
    if not os.path.exists(embedding_path):
        print(f"⚠️ ERROR: Embedding file not found locally. Please run 'modal volume get...' first.")
        return pd.DataFrame() 

    # Note: If the file is empty or corrupted, this read will fail. Using a defensive try/except.
    try:
        emb_df = pd.read_csv(embedding_path, index_col='Date', parse_dates=True)
    except pd.errors.EmptyDataError:
        print("⚠️ ERROR: Embedding file is empty. Skipping PCA.")
        return pd.DataFrame()

    # Run PCA
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(emb_df.fillna(0)) # Fillna 0 for safety before PCA
    
    # Create output DataFrame
    pca_df = pd.DataFrame(data=principal_components, index=emb_df.index)
    pca_df.columns = [f'Semantic_PCA_{i+1}' for i in range(num_components)]
    
    print(f"    PCA Variance Explained: {pca.explained_variance_ratio_.sum():.2%}")
    return pca_df

# --- Helper Function: Process Sector Index (Nifty Bank) ---
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

    # Check if news_df was successfully loaded before passing to process_sentiment
    if news_df is None:
        print("Error: News data failed to load.")
        return
        
    sentiment_series = process_sentiment(news_df, TRAIN_START, TEST_END)
    sector_df = process_sector_index()
    
    # --- NEW: Process Embeddings ---
    pca_df = process_semantic_embeddings(num_components=10)
    # -------------------------------

    # FIX 2: Initialize pca_cols outside the loop
    pca_cols = []
    if not pca_df.empty:
        pca_cols = [col for col in pca_df.columns if 'Semantic_PCA' in col]

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
        
        # C. Merge Sentiment
        df = df.join(sentiment_series['Sentiment_Decay'], how='left')
        df['Sentiment_Decay'].fillna(0, inplace=True)
        
        # --- NEW: Merge PCA Embeddings ---
        if not pca_df.empty:
            df = df.join(pca_df, how='left')
            # Fill NaN for days without news (Semantic_PCA_x) with zero vector
            df[pca_cols] = df[pca_cols].fillna(0.0) 
            print(f"   Merged {len(pca_cols)} PCA Semantic Features.")
        # ----------------------------------
        
        # --- Merge Sector Context and Calculate Beta ---
        if sector_df is not None: 
            # Indentation fixed here
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
            'Sector_Ret', 'Sector_Vol', 'Rolling_Beta'
        ]
        # Dynamically add PCA features to scaling list
        # FIX 3: Corrected indentation for extend() to be outside the 'if' block.
        if pca_cols:
            features_to_scale.extend(pca_cols) 

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