import numpy as np
import pandas as pd
from numpy import linalg
from datetime import datetime, timedelta
from scipy.linalg import inv as inv
from typing import List, Dict, Union


class BoardMarketIndex:
    def __init__(self, prices_csv_name: str, momentum_csv_name: str, trading_date: str,
                 leader_search_by: str = "highest_momentum_score",
                 correlation_threshold_for_grp: float = 0.5, ma_period: int = 60):
        """
        Initialize the BoardMarketIndex class for ETF market analysis.

        Args:
            prices_csv_name (str): Path to CSV file containing ETF price data
            momentum_csv_name (str): Path to CSV file containing momentum scores
            trading_date (str): Date in 'YYYY-MM-DD' format for analysis standpoint
            leader_search_by (str): Method to select group leaders ('highest_momentum_score' or 'highest_abs_correlation')
            correlation_threshold_for_grp (float): Minimum correlation for grouping (default: 0.6)
            ma_period (int): Moving average period for market breadth (default: 60)

        Attributes:
            etf_file_name: CSV file name containing ETF prices
            momentum_file_name: CSV file name containing momentum scores
            standing_point_date: Date for analysis standpoint
            ticker_list: List of ETF tickers to analyze (read from prices_csv_name)
            correlation_threshold_for_grp: Minimum correlation for grouping
            ma_period: Moving average period for market breadth
            prices_df: Filtered price DataFrame
            groups: Dictionary of asset groups
            marketbreadth: Market breadth analysis DataFrame
        """
        self.etf_file_name = prices_csv_name
        self.momentum_file_name = momentum_csv_name
        self.standing_point_date = trading_date
        self.leader_search_by = leader_search_by
        self.correlation_threshold_for_grp = correlation_threshold_for_grp
        self.ma_period = ma_period

        # Read the ticker list from the columns of the CSV file
        initial_list_price_df = pd.read_csv(self.etf_file_name, index_col=0)
        self.ticker_list = list(initial_list_price_df.columns)

        # Initialize with function calls
        self.prices_df = self.get_filtered_data()
        self.groups = self.construct_group_withAsset()
        self.marketbreadth = self.get_group_marketbreadth()

    def get_filtered_data(self) -> pd.DataFrame:
        """Get filtered ETF price data based on standing point date and ticker list."""
        initial_list_price_df = pd.read_csv(self.etf_file_name, index_col=0)
        selected_etf_prices = initial_list_price_df[self.ticker_list]
        selected_etf_prices = selected_etf_prices[
            selected_etf_prices.index <= self.standing_point_date]

        standing_point = datetime.strptime(self.standing_point_date, "%Y-%m-%d")
        listing_before_date_str = (standing_point - timedelta(days=365)).strftime("%Y-%m-%d")
        listing_before_date_dt = pd.to_datetime(listing_before_date_str)
        while listing_before_date_str not in selected_etf_prices.index:
            listing_before_date_dt -= pd.Timedelta(days=1)
            listing_before_date_str = listing_before_date_dt.strftime('%Y-%m-%d')

        print("Filtering IPO-too-late etf...")
        filtered_df = selected_etf_prices.loc[:, selected_etf_prices.loc[listing_before_date_str].notna()]
        print("Filtering delisted etf...")
        filtered_df = filtered_df.loc[:, ~filtered_df.iloc[-1].isna()].dropna()
        return filtered_df

    def construct_group_withAsset(self) -> Dict:
        """Construct groups of assets based on correlation with configurable leader selection."""
        correlation_matrix = self.prices_df.corr()

        # Determine group leaders based on the specified method
        if self.leader_search_by == "highest_momentum_score":
            momentum_df = pd.read_csv(self.momentum_file_name, index_col=0)
            # Filter momentum scores for the trading date
            if self.standing_point_date in momentum_df.index:
                momentum_scores = momentum_df.loc[self.standing_point_date]
                ranked_assets = pd.DataFrame({
                    'Asset': momentum_scores.index,
                    'Momentum_Score': momentum_scores.values
                }).sort_values(by='Momentum_Score', ascending=False)
                # Filter to only include assets that are in filtered_df
                ranked_assets = ranked_assets[ranked_assets['Asset'].isin(self.prices_df.columns)]
            else:
                raise ValueError(f"No momentum scores found for {self.standing_point_date}")
        elif self.leader_search_by == "highest_abs_correlation":
            abs_sums = correlation_matrix.abs().sum()
            ranked_assets = abs_sums.sort_values(ascending=False).reset_index()
            ranked_assets.columns = ['Asset', 'Abs_Correlation_Sum']
        else:
            raise ValueError("leader_search_by must be 'highest_momentum_score' or 'highest_abs_correlation'")

        def group_assets(correlation_matrix, leader_asset, threshold):
            grouped_assets = [leader_asset]
            # Only iterate over assets that are in the correlation matrix (i.e., in filtered_df)
            for other_asset in correlation_matrix.columns:
                if other_asset != leader_asset and correlation_matrix.loc[leader_asset, other_asset] >= threshold:
                    grouped_assets.append(other_asset)
            return grouped_assets

        groups = {}
        used_assets = set()

        for _, row in ranked_assets.iterrows():
            grp_leader_asset = row['Asset']
            if grp_leader_asset not in used_assets:
                group = group_assets(correlation_matrix, grp_leader_asset, self.correlation_threshold_for_grp)
                if len(group) > 0:
                    groups[grp_leader_asset] = group
                    used_assets.update(group)

        return groups

    def get_group_marketbreadth(self) -> pd.DataFrame:
        """Calculate market breadth for each group, avoiding PerformanceWarning with pd.concat."""
        ma_df = self.prices_df.rolling(window=self.ma_period).mean()
        combined_df = self.prices_df.join(ma_df, rsuffix='_MA', how='inner').dropna()

        # Precompute Above_MA columns efficiently using pd.concat
        above_ma_cols = {
            f'{asset}_Above_MA': (combined_df[asset] > combined_df[f'{asset}_MA']).astype(bool)
            for asset in self.prices_df.columns
        }
        above_ma_df = pd.concat(above_ma_cols, axis=1)
        combined_df = pd.concat([combined_df, above_ma_df], axis=1)

        momentum_indices = {}
        for start_asset, group in self.groups.items():
            above_ma_count = 0
            total_assets = len(group)
            for asset in group:
                above_ma_count += combined_df[f'{asset}_Above_MA'].iloc[-1]
            momentum_index = (above_ma_count / total_assets) * 100 if total_assets > 0 else 0
            momentum_indices[start_asset] = {
                'Momentum_Index': momentum_index,
                'Assets': group
            }

        return pd.DataFrame(momentum_indices).T

    def asset_is_in_strong_momentum_group(self, specific_asset: str) -> bool:
        """Check if an asset is in a strong momentum group (Momentum_Index > 50)."""
        high_momentum_groups = self.marketbreadth[self.marketbreadth['Momentum_Index'] > 50]
        return any(specific_asset in group for group in high_momentum_groups['Assets'])

    def check_asset_group(self, asset: str) -> List:
        """Return the group containing the specified asset."""
        for group_leader, assets in self.groups.items():
            if asset in assets:
                return assets
        return []


if __name__ == "__main__":
    # Example usage
    etf_file_name = "ETFs_daily_prices.csv"
    momentum_file_name = "average_momentum_returns.csv"
    trading_date = "2020-03-31"

    # Initialize the class
    market_index = BoardMarketIndex(
        prices_csv_name=etf_file_name,
        momentum_csv_name=momentum_file_name,
        trading_date=trading_date,
        leader_search_by="highest_momentum_score",
        correlation_threshold_for_grp=0.5,
        ma_period=60
    )

    # Print some results
    print("\nFiltered Prices DataFrame:")
    print(market_index.prices_df.head())

    print("\nAsset Groups:")
    for leader, group in market_index.groups.items():
        print(f"{leader}: {group}")

    print("\nMarket Breadth:")
    print(market_index.marketbreadth)

    # Check specific asset
    test_asset = "AGG"
    if market_index.asset_is_in_strong_momentum_group(test_asset):
        print(f"\n{test_asset} is in a strong momentum group!")
    else:
        print(f"\n{test_asset} is not in a strong momentum group. Look for next top momentum ETFs.")

    # Check which group an asset belongs to
    asset_group = market_index.check_asset_group(test_asset)
    print(f"\nGroup containing {test_asset}: {asset_group}")