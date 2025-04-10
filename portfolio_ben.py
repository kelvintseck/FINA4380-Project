import pandas as pd
import numpy as np
import os
from typing import Dict, Union, Optional, Tuple
from datetime import datetime
from IPython.display import display

from scipy.stats import norm

# Global variables (to be included in the config.py)
ETF_FILE_NAME = "ETFs_daily_prices.csv"
RF_FILE_NAME = "riskFree.csv"
W_FILE_NAME = "weight_SR_MU_AbsCorr_GrpCorr0.7.csv"


class Portfolio:
    """
    A class to manage an evolving portfolio with manual rebalancing, transaction costs,
    and separation of visible/hidden price data to prevent look-ahead bias.
    
    Attributes:
        __prices (pd.DataFrame): Hidden full price history (start to end), loaded from CSV.
        visible_prices (pd.DataFrame): Prices up to the current date only.
        current_date (pd.Timestamp): Current simulation date.
        positions (pd.DataFrame): Historical shares held per asset.
        cash (pd.Series): Historical cash balance.
        value (pd.Series): Historical total portfolio value.
        returns (pd.Series): Historical daily returns.
        weights (pd.DataFrame): Historical target weights.
        init_cash (float): Initial cash amount (100 million).
        transaction_cost (float): Cost per trade as a proportion of trade value (e.g., 0.001 = 0.1%).
    """
    
    def __init__(self, etf_path: str, rf_path: str, start_date: str, transaction_cost: float = 0.001):
        """
        Initialize the Portfolio with ETF prices and risk-free rates.

        Args:
            etf_path: Path to the ETF prices CSV file.
            rf_path: Path to the daily annualized risk-free rate CSV file.
            start_date: Starting date for the simulation (e.g., "2018-09-28").
            transaction_cost: Transaction cost rate (default 0.001 = 0.1%).
        """
        # Load hidden price data from ETF CSV and RF CSV
        self.__prices = pd.read_csv(etf_path, index_col=0, parse_dates=True)
        self.__riskfree = pd.read_csv(rf_path, index_col=0, parse_dates=True)
        # Ensure the index is in datetime format
        self.__prices.index = pd.to_datetime(self.__prices.index)
        self.__riskfree.index = pd.to_datetime(self.__riskfree.index, format="%d/%m/%Y")
        # Trim risk-free rates to match ETF prices' start date
        self.__riskfree = self.__riskfree.loc[self.__prices.index.min():]
    
        
        # Convert user-specified start_date to datetime
        start_date = pd.to_datetime(start_date)
        if start_date < self.__prices.index[0]:
            raise ValueError(f"Start date {start_date.date()} not found in price data. The earliest date is {self.__prices.index[0].date()}")
        if start_date > self.__prices.index[-1]:
            raise ValueError(f"Start date {start_date.date()} not found in price data. The latest date is {self.__prices.index[1-1].date()}")
        if start_date not in self.__prices.index:
            start_date = self.__prices.index[self.__prices.index > start_date].min()
            print(f"Set start date to the next nearest trading date: {start_date.date()}")
        
        # Initialize visible prices and riskfree rates
        self.visible_prices = self.__prices.loc[:start_date].copy()
        self.visible_rf = self.__riskfree.loc[:start_date].copy()
        
        # Set current_date to user-specified date
        self.current_date = start_date
        self.init_cash = 100_000_000  # 100 million USD
        self.transaction_cost = transaction_cost
        
        # Initialize tracking (appendable)
        self.positions = pd.DataFrame(index=[self.current_date], columns=self.__prices.columns, data=0.0, dtype='float64')
        self.cash = pd.Series(index=[self.current_date], data=self.init_cash, dtype='float64')
        self.value = pd.Series(index=[self.current_date], data=self.init_cash, dtype='float64')
        self.returns = pd.Series(index=[self.current_date], data=0.0, dtype='float64')
        self.weights = pd.DataFrame(index=[self.current_date], columns=self.__prices.columns, data=0.0, dtype='float64')
        

    def advance_date(self, days: int = 1) -> None:
        """
        Move the simulation forward by the specified number of days (default is 1), 
        updating visible prices, value, returns, and weights based on price fluctuations.

        Args:
            days: Number of days to advance (default is 1).

        Raises:
            ValueError: If advancing exceeds the available price data.
        """
        for _ in range(days):
            current_idx = self.__prices.index.get_loc(self.current_date)
            new_idx = current_idx + 1
            if new_idx >= len(self.__prices):
                raise ValueError(f"Cannot advance; exceeds available price data. (Last day: {self.__prices.index[current_idx].date()})")
            
            self.current_date = self.__prices.index[new_idx]
            self.visible_prices.loc[self.current_date] = self.__prices.loc[self.current_date]
            self.visible_rf.loc[self.current_date] = self.__riskfree.loc[self.current_date]
            
            # Update portfolio state based on price changes
            current_prices = self.visible_prices.loc[self.current_date]
            asset_value = (self.positions.iloc[-1] * current_prices).sum()
            new_cash = (self.cash.iloc[-1] * 
                       self.visible_rf.loc[self.current_date]["Compounded Value"] / 
                       self.visible_rf.loc[:self.current_date]["Compounded Value"].shift(1).loc[self.current_date])   # cash get compounded 
            new_value = new_cash + asset_value
            new_returns = (new_value / self.value.iloc[-1]) - 1 if len(self.value) > 1 else 0.0
            
            # Update weights for all assets, even those with zero positions
            new_weights = pd.Series(index=self.weights.columns, data=0.0)  # Start with all 0s
            if new_value > 0:
                for asset in self.positions.columns:
                    new_weights[asset] = (self.positions.iloc[-1][asset] * current_prices[asset]) / new_value
                new_weights["cash"] = new_cash / new_value
            
            # Append updated state
            self.positions.loc[self.current_date] = self.positions.iloc[-1]
            self.cash.loc[self.current_date] = new_cash
            self.value.loc[self.current_date] = new_value
            self.returns.loc[self.current_date] = new_returns
            self.weights.loc[self.current_date] = new_weights.fillna(0.0)  # Ensure no NaN


    def calculate_transaction_cost(self, new_weights: Dict[str, float], is_rebalance: bool = False) -> Union[float, Tuple[pd.Series, float, pd.Series, float, float]]:
        """
        Calculate the transaction cost for rebalancing to new target weights.

        Args:
            new_weights: Dictionary of target weights for assets and optionally cash 
                         (e.g., {"GLD": 0.3, "XLY": 0.6, "cash": 0.1}).
            is_rebalance: Boolean indicating if the method is called by rebalance(). 
                          If True, returns additional data for rebalancing; if False, returns only the cost.
                          Defaults to False.

        Returns:
            If is_rebalance is False:
                float: Total transaction cost in dollars.
            If is_rebalance is True:
                Tuple containing:
                - pd.Series: Current prices of assets at the rebalancing date.
                - float: Current total portfolio value (cash + asset value).
                - pd.Series: New positions (shares) for each asset after rebalancing.
                - float: Target cash weight specified in new_weights (0.0 if not provided).
                - float: Total transaction cost in dollars.

        Raises:
            ValueError: If the sum of weights exceeds 1.
        """
        total_weight = sum(new_weights.values())
        if not total_weight <= 1:
            raise ValueError("Weights must sum to 1 or less.")
        
        # Current portfolio value
        current_prices = self.visible_prices.loc[self.current_date]
        current_value = self.cash.iloc[-1] + (self.positions.iloc[-1] * current_prices).sum()
        
        # Separate cash and asset weights
        cash_weight = new_weights.get("cash", 0.0)  # Default to 0 if not specified
        asset_weights = {k: v for k, v in new_weights.items() if k != "cash"}
        asset_weight_sum = sum(asset_weights.values())
        if asset_weight_sum > 0:
            asset_weights = {k: v / asset_weight_sum * (1 - cash_weight) for k, v in asset_weights.items()}
        
        # Calculate target values and new positions
        target_values = pd.Series(asset_weights) * current_value
        new_positions = target_values / current_prices
        
        # Calculate the change in positions from the current state
        position_change = new_positions - self.positions.iloc[-1]
        trade_value = (position_change.abs() * current_prices).sum()
        cost = trade_value * self.transaction_cost
        
        if is_rebalance:
            return current_prices, current_value, new_positions, cash_weight, cost
        return cost


    def rebalance(self, new_weights: Dict[str, float]) -> None:
        """
        Rebalance the portfolio to new target weights at the current date, 
        updating positions, cash, and weights while keeping total value constant (adjusted for costs).

        Args:
            new_weights: Dictionary of target weights for assets and optionally cash 
                         (e.g., {"GLD": 0.3, "XLY": 0.6, "cash": 0.1}).

        Raises:
            ValueError: If the sum of weights exceeds 1.
        """
        total_weight = sum(new_weights.values())
        if not total_weight <= 1:
            raise ValueError("Weights must sum to 1 or less.")
        
        current_prices, current_value, new_positions, cash_weight, cost = self.calculate_transaction_cost(
            new_weights, is_rebalance=True
        )
        
        asset_value = (new_positions * current_prices).sum()
        new_cash = current_value - asset_value - cost
        if new_cash < cash_weight * current_value:
            shortfall = cash_weight * current_value - new_cash
            asset_value -= shortfall
            # new_positions *= (asset_value / (new_positions * current_prices).sum())
            """---------------------------------------------------------------------------"""
            denominator = (new_positions * current_prices).sum()
            if denominator == 0:
                raise ValueError(f"Error: Division by zero encountered. Denominator value: {new_positions} & {current_prices}")
            else:
                new_positions *= (asset_value / denominator)
            """---------------------------------------------------------------------------"""
            new_cash = cash_weight * current_value
        
        # Value remains the same as before rebalancing, adjusted for costs
        # Returns not updated here (no time has passed)
        # Update state, ensuring all columns are filled with 0 where no data
        
        self.positions.loc[self.current_date] = new_positions.astype(float).fillna(0.0)
        self.cash.loc[self.current_date] = new_cash
        self.weights.loc[self.current_date] = pd.Series(new_weights).reindex(self.weights.columns, fill_value=0.0) # Reflects the new allocation
        

    def get_current_data(self) -> Dict[str, Union[pd.DataFrame, pd.Series, float]]:
        """
        Return the latest portfolio data visible to the user.
        """
        return self.get_history(days = 1)
        
        
    def get_history(self, days: int = 252)  -> Dict[str, Union[pd.DataFrame, pd.Series, float]]:
        """
        Return the historical portfolio data visible to the user. (Default length is 252 days)
        """
        return {
            "prices": self.visible_prices.copy().tail(days),
            "positions": self.positions.copy().tail(days),
            "cash": self.cash.copy().tail(days),
            "value": self.value.copy().tail(days),
            "returns": self.returns.copy().tail(days),
            "weights": self.weights.copy().tail(days),
            "riskfree rates": self.visible_rf.copy().tail(days)
        }


    # Performance Metrics
    def compute_daily_metrics(self, days: int = 252, alpha: float = 0.05) -> pd.DataFrame:
        """
        Compute comprehensive daily performance metrics including:
        - Cumulative Return
        - Annualized Return
        - Annualized Volatility (trailing 252D)
        - Annualized Sharpe Ratio (trailing 252D)
        - Sortino Ratio (trailing 252D)
        - Drawdown (cumulative)
        - Max Drawdown (cumulative)
        - Calmar Ratio (trailing 36M = 756D)
        - Value-at-Risk (trailing 252D)
        - Expected Shortfall (trailing 252D)
        
        Parameters:
            days: int, trading days per year
            alpha: float, confidence level for VaR/ES (e.g., 0.05 for 95%)
        """
        if len(self.returns) < 1:
            return pd.DataFrame()
        
        metrics = pd.DataFrame(index=self.returns.index)
        
        # 1. Cumulative Return
        metrics['Cumulative Return'] = (self.value / self.value.iloc[0]) - 1
        
        # 2. Annualized Return
        days_elapsed = np.arange(1, len(self.returns) + 1) 
        metrics['Annualized Return'] = (1 + metrics['Cumulative Return']) ** (365 / days_elapsed) - 1
        
        # 3. Annualized Volatility (trailing 252D)
        metrics['Annualized Volatility'] = self.returns.rolling(window=days).std() * np.sqrt(days)
        
        # 4. Annualized Sharpe Ratio (trailing 252D)
        excess_returns = self.returns - (self.visible_rf["Rate"]/days)
        metrics['Sharpe Ratio'] = (
            excess_returns.rolling(days).mean() / 
            self.returns.rolling(days).std() * 
            np.sqrt(days)
        )
        
        # 5. Sortino Ratio (trailing 252D)
        downside_returns = np.minimum(self.returns, 0)
        metrics['Sortino Ratio'] = (
            (self.returns.rolling(days).mean() - self.visible_rf["Rate"]/days) / 
            downside_returns.rolling(days).std() * 
            np.sqrt(days)
        )
        
        # 6. Drawdown (cumulative)
        rolling_max = self.value.rolling(window=len(self.value), min_periods=1).max()
        metrics['Drawdown'] = self.value / rolling_max - 1
        
        # 7. Max Drawdown (cumulative)
        metrics['Max Drawdown'] = metrics['Drawdown'].rolling(window=len(self.value), min_periods=1).min()
        
        # 8. Calmar Ratio (trailing 36M = 756D)
        metrics['Calmar Ratio'] = (
            metrics['Annualized Return'].rolling(756).mean() / 
            metrics['Max Drawdown'].rolling(756).min().abs()
        )
        
        # 9. Value-at-Risk (trailing 252D)
        metrics['VaR'] = self.returns.rolling(days).quantile(alpha)
        
        # 10. Expected Shortfall (trailing 252D)
        def calc_es(series):
            if len(series.dropna()) < 5:
                return np.nan
            return series[series <= series.quantile(alpha)].mean()
        
        metrics['Expected Shortfall'] = self.returns.rolling(days).apply(calc_es)
        
        return metrics
    
    


if __name__ == "__main__":   # For testing and debugging
    folder_path = os.path.dirname(__file__)   # Put this .py file together with .csv files
    file_ETFs = os.path.join(folder_path, ETF_FILE_NAME)   # Contain the historical prices of our stocks universe
    file_rf = os.path.join(folder_path, RF_FILE_NAME)   # Contain the historical annualized riskfree rates
    file_weight = os.path.join(folder_path, W_FILE_NAME)

    portfolio = Portfolio(file_ETFs, file_rf, start_date = "2015-10-08", transaction_cost = 0.001)
    weights = pd.read_csv(file_weight, index_col=0)[1500:].to_dict(orient='list') # start to have valid data (non 0 weights) around 1500
    for i in range(len(weights["cash"])):
        cur_weight = {}
        for key in weights.keys():
            cur_weight[key] = weights[key][i]
        portfolio.rebalance(cur_weight)
        portfolio.advance_date()

    # for i in range(2000):
    #     # rnt1 = np.random.uniform(0,1)
    #     # rnt2 = np.random.uniform(0, 1-rnt1)
    #     # print(rnt1, rnt2, 1-rnt1-rnt2)
    #     # curr_data = portfolio.get_current_data()
    #     # print(curr_data["weights"])
    #     # portfolio.rebalance({"GLD": rnt1, "XLY": rnt2, "cash": 1-rnt1-rnt2})
    #
    #     portfolio.rebalance({"EWC": np.random.uniform(0,0.3), "SUSA": np.random.uniform(0,0.3), "IJH": np.random.uniform(0,0.3)})
    #     portfolio.advance_date()
        
    
    all = portfolio.get_history()
    
    for key, df in all.items():
        print("="*40 + f"  {key}  " + "="*40)
        with pd.option_context('display.float_format', '{:.10}'.format):
            display(df)
    
    
    performance = portfolio.compute_daily_metrics()
    display(performance)
    
    
    
    from visualization import create_portfolio_dashboard
    create_portfolio_dashboard(portfolio, performance, os.path.join(folder_path, "portfolio_evaluation.html"))
    
    
    
    