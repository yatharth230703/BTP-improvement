import pandas as pd
import numpy as np

def process_sentiment(news_df, start_date, end_date):
    """
    Aggregates news sentiment and applies decay function.
    """
    # Filter for date range
    mask = (news_df['Date'] >= start_date) & (news_df['Date'] <= end_date)
    df = news_df.loc[mask].copy()
    
    # 1. Aggregation: Daily Average Sentiment
    daily_sentiment = df.groupby('Date')['sentiment_score'].mean().reset_index()
    daily_sentiment.set_index('Date', inplace=True)
    
    # Reindex to ensure continuous timeline (fill missing days with 0)
    idx = pd.date_range(start_date, end_date)
    daily_sentiment = daily_sentiment.reindex(idx, fill_value=0)
    
    # 2. Decay Function (Memory)
    # E_t = S_t + lambda * E_{t-1}
    decay_factor = 0.85
    decayed_scores = []
    current_score = 0
    
    for raw_score in daily_sentiment['sentiment_score']:
        current_score = raw_score + (decay_factor * current_score)
        decayed_scores.append(current_score)
        
    daily_sentiment['Sentiment_Decay'] = decayed_scores
    if daily_sentiment.index.tz is not None:
        daily_sentiment.index = daily_sentiment.index.tz_localize(None)
    return daily_sentiment