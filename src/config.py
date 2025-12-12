import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PLOT_DIR = os.path.join(PROCESSED_DATA_DIR, 'visualization_plots')

# Create dirs if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Date Limits (Strictly excluding 2020)
TRAIN_START = '2015-01-01' # Adjust based on how far back you want to go
TRAIN_END = '2018-12-31'
TEST_START = '2019-01-01'
TEST_END = '2019-12-31' 
# Any data >= 2020-01-01 is dropped

# Stock List
# Stock List
TARGET_STOCKS = ['SBIN', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'PNB']