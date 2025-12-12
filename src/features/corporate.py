import pandas as pd

def apply_corporate_actions(price_df, corporate_df, ticker):
    """
    Adjusts prices for splits and flags dividends.
    """
    price_df = price_df.copy()
    
    # Filter actions for this specific ticker
    # Assuming corporate_df has columns ['Date', 'Ticker', 'Action', 'Ratio/Value']
    actions = corporate_df[corporate_df['Ticker'] == ticker].copy()
    actions['Date'] = pd.to_datetime(actions['Date'])
    
    # Create Flag Columns
    price_df['Is_Split'] = 0
    price_df['Is_Dividend'] = 0
    
    for index, row in actions.iterrows():
        event_date = row['Date']
        if event_date in price_df.index:
            if row['Action'] == 'Split':
                price_df.at[event_date, 'Is_Split'] = 1
                # Apply Split Adjustment (Simplified: Retroactive division)
                ratio = float(row['Value']) # e.g., 10 for 10:1 split
                # In real backtesting, we adjust historical prices:
                mask = price_df.index < event_date
                price_df.loc[mask, ['Open', 'High', 'Low', 'Close']] /= ratio
                price_df.loc[mask, 'Volume'] *= ratio
                
            elif row['Action'] == 'Dividend':
                price_df.at[event_date, 'Is_Dividend'] = 1
                
    return price_df