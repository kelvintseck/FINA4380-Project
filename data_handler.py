import pandas as pd
import yfinance as yf
from typing import List, Dict, Union   # for type hints in function inputs, improve readability and collaboration efficiency

    """
    Main Purpose: To handle the acquisition and preparation of stock data, including downloading CSV files and calculating technical indicators. This file covers steps 1 (downloading data) and 2 (data processing) of your process.
    
    My opinions: 
    - There should be several sub-functions in Stock Class (for different kinds of technical indicators)
    - We should run sub-functions with a certain frequency for all stocks data. (e.g. per month). 
    - After running the sub-functions, update the new values in the self.indicators.
    
    Below are just some sample code (which may not be correct), feel free to change it, but it's better to make it precise (with type hints and doc string)
    """


class Stock:
    """
    Represents an individual stock with its symbol and indicator data.
    """
    def __init__(self, symbol: str, data: pd.DataFrame, date1: str, date2: str):
        """
        Initialize the Stock object with raw data for a specific date range.
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            data: Raw DataFrame with stock data (e.g., from yfinance)
            date1: Start date in 'YYYY-MM-DD' format
            date2: End date in 'YYYY-MM-DD' format
        """
        self.symbol = symbol
        # Convert date strings to datetime for slicing
        start_date = pd.to_datetime(date1)
        end_date = pd.to_datetime(date2)
        # Slice the data between date1 and date2, inclusive
        self.indicators = data[['Close', 'Volume']].loc[start_date:end_date]
        # Store the number of periods for later use in updates


    def calculate_sma(self, window: int = 20) -> None:
        """
        Calculate Simple Moving Average and add it to indicators.
        Args:
            window: Number of periods for the moving average
        """
        self.indicators[f'SMA{window}'] = self.indicators['Close'].rolling(window=window, min_periods=1).mean()


    def calculate_rsi(self, window: int = 14) -> None:
        """
        Calculate Relative Strength Index and add it to indicators.
        Args:
            window: Number of periods for RSI calculation
        """
        delta = self.indicators['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        rs = rs.replace([float('inf'), -float('inf')], 0)  # Handle division by zero
        self.indicators[f'RSI{window}'] = 100 - (100 / (1 + rs))


    def calculate_volume_ma(self, window: int = 20) -> None:
        """
        Calculate Volume Moving Average and add it to indicators.
        Args:
            window: Number of periods for the volume moving average
        """
        self.indicators[f'VolumeMA{window}'] = self.indicators['Volume'].rolling(window=window, min_periods=1).mean()


    def update_indicators(self, new_row: Dict[str, float]) -> None:
        """
        Update indicators by removing the oldest row and adding a new row.
        Args:
            new_row: Dictionary with new data (e.g., {'Close': 185.0, 'Volume': 5500000})
        """
        # My opinion: input the new enddate, calculate new indicators for the lastest day, delete the oldest day
        # Ensure new_row has the required base columns
        required_columns = ['Close', 'Volume']
        for col in required_columns:
            if col not in new_row:
                raise ValueError(f"New row must include '{col}'")

        # Create a new DataFrame row with the next index
        new_index = self.indicators.index[-1] + pd.Timedelta(days=1)  # Assume daily data
        new_df_row = pd.DataFrame([new_row], index=[new_index], columns=self.indicators.columns)

        # Remove the oldest row and append the new row
        self.indicators = pd.concat([self.indicators.iloc[1:], new_df_row])

        # Recalculate all indicators
        self.calculate_sma()
        self.calculate_rsi()
        self.calculate_volume_ma()



def download_stock_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads stock data for given symbols and date range.
    Input:
        symbols: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    Output:
        DataFrame with stock data (columns: Date, Open, High, Low, Close, Volume, etc.)
    """
    data = yf.download(symbols, start=start_date, end=end_date)
    return data


def save_data_to_csv(data: pd.DataFrame, filename: str) -> None:
    """
    Saves stock data to a CSV file.
    Input:
        data: DataFrame containing stock data
        filename: Name of the file to save (e.g., 'stock_data.csv')
    Output:
        None
    """
    data.to_csv(filename)
    


if __name__ == '__main__':  # For testing and debugging
    pass