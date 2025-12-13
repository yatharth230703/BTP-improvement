# src/utils/backtester.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class VectorizedBacktester:
    def __init__(self, ticker, actual_prices, predicted_returns, dates, transaction_cost=0.001):
        """
        args:
            actual_prices (array): The raw close prices (for Buy & Hold comparison).
            predicted_returns (array): The Log Returns predicted by your model.
            dates (array): Timestamps for the x-axis.
            transaction_cost (float): Cost per trade (e.g., 0.1% = 0.001).
        """
        self.ticker = ticker
        self.data = pd.DataFrame({
            'Price': actual_prices,
            'Pred_Signal': predicted_returns
        }, index=dates)
        
        # Calculate Actual Log Returns for the PnL calculation
        self.data['Actual_Return'] = np.log(self.data['Price'] / self.data['Price'].shift(1))
        self.cost = transaction_cost

    def run_strategy(self, threshold=0.0005):
        """
        Runs a Long-Short strategy based on the signal strength.
        """
        df = self.data.copy()
        
        # --- STRATEGY LOGIC ---
        # 1. Define Position
        # If model says return > 0.05% -> Buy (+1)
        # If model says return < -0.05% -> Sell (-1)
        # Otherwise -> Cash (0)
        df['Position'] = 0
        df.loc[df['Pred_Signal'] > threshold, 'Position'] = 1
        df.loc[df['Pred_Signal'] < -threshold, 'Position'] = -1
        
        # 2. Shift Position
        # We compute signal at close of day 't', so we enter position at 't+1'
        df['Position'] = df['Position'].shift(1)
        
        # 3. Calculate Strategy Returns
        # Strategy = Position * Market_Return
        df['Strat_Return'] = df['Position'] * df['Actual_Return']
        
        # 4. Transaction Costs
        # We pay cost whenever Position changes (e.g., 0 to 1, or 1 to -1)
        df['Trades'] = df['Position'].diff().abs()
        df['Strat_Return_Net'] = df['Strat_Return'] - (df['Trades'] * self.cost)
        
        # 5. Cumulative Returns (Equity Curve)
        df['Cum_BnH'] = df['Actual_Return'].cumsum().apply(np.exp) # Buy and Hold
        df['Cum_Strat'] = df['Strat_Return_Net'].cumsum().apply(np.exp) # Your Model
        
        self.results = df.dropna()
        return self.results

    def plot_equity_curve(self, save_dir):
        if not hasattr(self, 'results'):
            raise Exception("Run strategy first!")
            
        df = self.results
        
        # Calculate Metrics
        total_ret = (df['Cum_Strat'].iloc[-1] - 1) * 100
        bnh_ret = (df['Cum_BnH'].iloc[-1] - 1) * 100
        sharpe = (df['Strat_Return_Net'].mean() / df['Strat_Return_Net'].std()) * np.sqrt(252)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Cum_BnH'], label=f'Buy & Hold ({bnh_ret:.1f}%)', color='grey', alpha=0.5, linestyle='--')
        plt.plot(df.index, df['Cum_Strat'], label=f'Trimodal Model ({total_ret:.1f}%)', color='green', linewidth=2)
        
        plt.title(f"💰 PnL Backtest: {self.ticker} (Sharpe: {sharpe:.2f})")
        plt.ylabel("Portfolio Value ($1 start)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        path = f"{save_dir}/{self.ticker}_backtest.png"
        plt.savefig(path)
        plt.close()
        print(f"   💰 Backtest saved to {path}")
        return path