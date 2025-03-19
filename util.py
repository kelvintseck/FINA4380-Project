import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
    """
    To store the global variables and functions that other .py files may need to access etc.
    """


# --- Configuration Settings ---
DEFAULT_START_DATE = '2024-01-01'
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')  # Current date
DATE_FORMAT = '%Y-%m-%d'
INITIAL_CAPITAL = 10000.0
TRADING_DAYS_PER_YEAR = 252  # For annualized metrics like Sharpe Ratio

    
if __name__ == '__main__':  # For testing and debugging
    pass