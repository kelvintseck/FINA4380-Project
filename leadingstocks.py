import yfinance as yf
import pandas as pd
from datetime import datetime

# Define the time range: January 1950 to December 2014
start_date = "1950-01-01"
end_date = "2014-12-31"

# Example list of 100 stock tickers (replace with your actual list)
tickers = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK.B', 'JPM',
    'UNH', 'V', 'MA', 'HD', 'PG', 'JNJ', 'COST', 'WMT', 'XOM', 'CVX',
    'DIS', 'NFLX', 'ORCL', 'CSCO', 'ADBE', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO',
    'PEP', 'KO', 'MRK', 'PFE', 'LLY', 'ABBV', 'TMO', 'DHR', 'ABT', 'MDT',
    'BAC', 'WFC', 'GS', 'MS', 'C', 'SCHW', 'BLK', 'AXP', 'SPGI', 'MCO',
    'CAT', 'DE', 'MMM', 'GE', 'HON', 'UPS', 'FDX', 'BA', 'LMT', 'RTX',
    'T', 'VZ', 'CMCSA', 'CHTR', 'SBUX', 'MCD', 'YUM', 'DPZ', 'NKE', 'TJX',
    'LOW', 'TGT', 'DG', 'DLTR', 'STZ', 'SYY', 'GIS', 'KHC', 'MO', 'PM',
    'DASH', 'TKO', 'WSM', 'EXE', 'PLTR', 'DELL', 'ERIE', 'CRM', 'NOW', 'PANW',
    'IBM', 'ACN', 'FIS', 'PAYX', 'CTAS', 'MOH', 'TSN', 'SJM', 'EXC', 'TAP'
]  # Placeholder for illustration

# Ensure we have exactly x tickers for this example
tickers = tickers[:100]

# Fetch monthly data for all tickers in one call
data = yf.download(tickers, start=start_date, end=end_date, interval="1mo", group_by="ticker")

# Extract the adjusted close price for each ticker
# If group_by="ticker", data is multi-level columned: (ticker, OHLCV)
adj_close = pd.DataFrame()
for ticker in tickers:
    if ticker in data.columns.levels[0]:
        adj_close[ticker] = data[ticker]['Close']
    else:
        adj_close[ticker] = pd.Series(index=data.index, dtype=float)  # NaN if no data

# Reset index to ensure dates are in a column (optional)
adj_close = adj_close.reset_index()
adj_close['Date'] = adj_close['Date'].dt.strftime('%Y-%m')  # Format as YYYY-MM

# Display the shape and a sample
print(f"Shape of DataFrame: {adj_close.shape}")
print(adj_close.head())

# Save to CSV
adj_close.to_csv('stock_monthly_prices_1950_2014.csv', index=False)