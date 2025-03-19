import pandas as pd
from typing import Dict, List

    """
    Main Purpose: To create and manage the investment portfolio based on specific criteria
    """

def select_stocks(data: pd.DataFrame, criteria: str) -> List[str]:
    """
    Selects stocks based on a specified criterion.
    Input:
        data: DataFrame with stock data and indicators (multi-index if multiple stocks)
        criteria: String defining selection rule (e.g., 'highest_rsi')
    Output:
        List of selected stock symbols
    """
    if criteria == 'highest_rsi':
        latest_rsi = data['RSI'].iloc[-1]  # Last row for each stock
        top_stocks = latest_rsi.nlargest(3).index.get_level_values('Ticker').tolist()  # Top 3 RSI
        return top_stocks
    return []


def allocate_weights(stocks: List[str], method: str) -> Dict[str, float]:
    """
    Allocates weights to selected stocks.
    Input:
        stocks: List of stock symbols
        method: String defining weighting method (e.g., 'equal')
    Output:
        Dictionary mapping stocks to their weights
    """
    if method == 'equal':
        weight = 1.0 / len(stocks)
        return {stock: weight for stock in stocks}
    return {}


def get_portfolio(data: pd.DataFrame, criteria: str, weighting_method: str) -> Dict[str, float]:
    """
    Combines stock selection and weighting to create a portfolio.
    Input:
        data: DataFrame with stock data and indicators
        criteria: String for stock selection rule
        weighting_method: String for weight allocation method
    Output:
        Dictionary of stocks and their weights
    """
    selected_stocks = select_stocks(data, criteria)
    portfolio = allocate_weights(selected_stocks, weighting_method)
    return portfolio

if __name__ == '__main__':  # For testing and debugging
    pass