import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import PLOT_DIR

def plot_price_vs_sentiment(df, ticker_name):
    """
    Dual axis plot: Price (Line) vs Sentiment Decay (Area)
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price (INR)', color=color)
    ax1.plot(df.index, df['Close'], color=color, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Sentiment (Decayed)', color=color)
    ax2.fill_between(df.index, df['Sentiment_Decay'], color=color, alpha=0.3, label='News Sentiment')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'{ticker_name}: Price Dynamics vs. News Sentiment')
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{ticker_name}_price_sentiment.png'))
    plt.close()

def plot_correlation_heatmap(df, ticker_name):
    """
    Heatmap to check feature redundancy
    """
    plt.figure(figsize=(10, 8))
    # Select numeric columns only
    corr = df[['Close', 'Log_Ret', 'RSI', 'MACD', 'Volatility', 'Sentiment_Decay']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'{ticker_name}: Feature Correlation Matrix')
    plt.savefig(os.path.join(PLOT_DIR, f'{ticker_name}_heatmap.png'))
    plt.close()