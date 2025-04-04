import numpy as np
import pandas as pd
from numpy import linalg
from datetime import datetime, timedelta
from scipy.linalg import inv as inv
from typing import List, Dict, Union


class BoardMarketIndex:
    def __init__(self, prices_csv_name: str, trading_date: str):
        """
        Initialize the BoardMarketIndex class for ETF market analysis.

        Args:
            prices_csv_name (str): Path to CSV file containing ETF price data
            trading_date (str): Date in 'YYYY-MM-DD' format for analysis standpoint

        Attributes:
            etf_file_name: CSV file name containing ETF prices
            standing_point_date: Date for analysis standpoint
            ticker_list: List of ETF tickers to analyze
            correlation_threshold_for_grp: Minimum correlation for grouping (default: 0.5)
            ma_period: Moving average period for market breadth (default: 60)
            prices_df: Filtered price DataFrame
            groups: Dictionary of asset groups
            marketbreadth: Market breadth analysis DataFrame
        """
        self.etf_file_name = prices_csv_name
        self.standing_point_date = trading_date
        self.ticker_list = ['VGSH', 'SCHO', 'BIL', 'GBIL', 'GOVT', 'SPTI', 'VGLT', 'VTIP', 'SCHP', 'LTPZ',
                            'VCSH', 'IGIB', 'HYG', 'BKLN', 'SRLN', 'SNLN', 'FLBL', 'BWX', 'IGOV', 'BNDX',
                            'EMB', 'EMLC', 'HYEM', 'EMHY', 'MUB', 'VTEB', 'TFI', 'MUNI', 'HYMB', 'SHYD',
                            'HYD', 'GLTR', 'SGOL', 'USO', 'UNG', 'DBE', 'DBA', 'CORN', 'WEAT', 'SOYB',
                            'CANE', 'DBB', 'JJM', 'COPX', 'REMX', 'UUP', 'FXY', 'FXB', 'CEW', 'BZF',
                            'FXCH', 'EMFX', 'VIXY', 'SVOL', 'VXXB', 'ETHE', 'ARKB', 'RPAR', 'DIVB', 'AOR',
                            'VEGI', 'MOO', 'XHB', 'VAW', 'DOW', 'FXZ', 'WOOD', 'CUT', 'FUND', 'JJU',
                            'FOIL', 'JJC', 'PICK', 'GDX', 'SIVR', 'SIL', 'PPLT', 'PALL', 'KOL', 'SLX',
                            'PSCL', 'CARZ', 'PBS', 'XRT', 'EMFM', 'PKG', 'XAR', 'PEJ', 'BITE', 'RTH',
                            'LUXE', 'ONLN', 'IBUY', 'FDIS', 'BETZ', 'JETS', 'TRYP', 'KCE', 'IAI', 'KRE',
                            'MORT', 'IXIS', 'FINX', 'KIE', 'REZ', 'SRET', 'XLRE', 'PBJ', 'FTXG', 'XLP',
                            'VICE', 'IBB', 'XBI', 'PJP', 'XPH', 'IHF', 'IHI', 'XHE', 'QCLN', 'ICLN',
                            'XLU', 'XOP', 'XLE', 'AMLP', 'CRAK', 'URA', 'ITA', 'XTN', 'IYT', 'SEA',
                            'IGV', 'WCLD', 'SMH', 'TAN']
        self.correlation_threshold_for_grp = 0.5
        self.ma_period = 60  # for market breadth standard

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
        """Construct groups of assets based on correlation."""
        correlation_matrix = self.prices_df.corr()
        positive_sums = correlation_matrix.clip(lower=0).sum()
        ranked_assets = positive_sums.sort_values(ascending=False).reset_index()
        ranked_assets.columns = ['Asset', 'Positive_Correlation_Sum']
        ranked_assets['Rank'] = ranked_assets['Positive_Correlation_Sum'].rank(ascending=False)

        def group_assets(correlation_matrix, leader_asset, threshold):
            grouped_assets = [leader_asset]
            for other_asset in correlation_matrix.columns:
                if other_asset != leader_asset and correlation_matrix.loc[leader_asset, other_asset] >= threshold:
                    grouped_assets.append(other_asset)
            return grouped_assets

        threshold = self.correlation_threshold_for_grp
        groups = {}
        used_assets = set()

        for _, row in ranked_assets.iterrows():
            grp_leader_asset = row['Asset']
            if grp_leader_asset not in used_assets:
                group = group_assets(correlation_matrix, grp_leader_asset, threshold)
                if len(group) > 0:
                    groups[grp_leader_asset] = group
                    used_assets.update(group)

        return groups

    def get_group_marketbreadth(self) -> pd.DataFrame:
        """Calculate market breadth for each group."""
        ma_df = self.prices_df.rolling(window=self.ma_period).mean()
        combined_df = self.prices_df.join(ma_df, rsuffix='_MA', how='inner').dropna()
        for asset in self.prices_df.columns:
            combined_df[f'{asset}_Above_MA'] = combined_df[asset] > combined_df[f'{asset}_MA']

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
    trading_date = "2011-03-31"

    # Initialize the class
    market_index = BoardMarketIndex(etf_file_name, trading_date)

    # Print some results
    print("\nFiltered Prices DataFrame:")
    print(market_index.prices_df.head())

    print("\nAsset Groups:")
    for leader, group in market_index.groups.items():
        print(f"{leader}: {group}")

    print("\nMarket Breadth:")
    print(market_index.marketbreadth)

    # Check specific asset
    test_asset = "AOR"
    if market_index.asset_is_in_strong_momentum_group(test_asset):
        print(f"\n{test_asset} is in a strong momentum group!")
    else:
        print(f"\n{test_asset} is not in a strong momentum group. Look for next top momentum ETFs.")

    # Check which group an asset belongs to
    asset_group = market_index.check_asset_group(test_asset)
    print(f"\nGroup containing {test_asset}: {asset_group}")
