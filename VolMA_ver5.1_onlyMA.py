import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import SPX_datadownload

"""
Version 2 (Modified):
- No volume signals; only price cuts below 200MA trigger a transition to 'undefined' in a bull market.
- Slope threshold only defines bull and bear states; undefined state is only triggered by price cuts below 200MA.
- Cooldown periods: bull=0, bear=0, undefined=30.
- Added state period duration to CSV output, reflecting the new cooldown periods.
- Added 'Blocked by Cooldown' column to CSV to indicate if a potential state change was blocked by the cooldown period.
"""

class VolumeIndicatorBacktest:
    def __init__(self, data: pd.DataFrame, ma_period: int = 200,
                 cooldown_periods: dict = None, slope_threshold: float = 0.001):
        """
        Initialize with historical data and calculate market trend, using price cuts below 200MA
        to trigger a transition to 'undefined' in a bull market.

        Parameters:
        data (pd.DataFrame): DataFrame with 'Volume' and 'Close' columns
        ma_period (int): Period for moving average to determine trend (default: 200)
        cooldown_periods (dict): Cooldown periods for each trend {'bull': int, 'bear': int, 'undefined': int}
        slope_threshold (float): Threshold for relative MA slope to determine trend (default: 0.001, i.e., 0.1%)
        """
        self.data = data.copy()
        # Ensure the index is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                raise ValueError(f"Could not convert DataFrame index to datetime format: {e}")

        # Store ma_period for use in other methods
        self.ma_period = ma_period

        # Calculate future returns
        self.data['return_10'] = self.data['Close'].pct_change(periods=10).shift(-10)
        self.data['return_20'] = self.data['Close'].pct_change(periods=20).shift(-20)

        # Calculate 200-day MA
        self.data['ma_200'] = self.data['Close'].rolling(window=ma_period).mean()
        # Calculate relative slope: (ma_200[t] - ma_200[t-1]) / ma_200[t-1]
        self.data['ma_slope'] = self.data['ma_200'].pct_change()

        # Inspect the typical range of ma_slope to determine an appropriate slope_threshold
        print("Descriptive statistics of ma_slope (relative change):")
        print(self.data['ma_slope'].describe())

        # Initialize trend column with string dtype
        self.data['trend'] = pd.Series(dtype='string')

        # Set cooldown periods for each trend
        if cooldown_periods is None:
            cooldown_periods = {'bull': 0, 'bear': 0, 'undefined': 30}  # Updated cooldown periods
        self.cooldown_periods = cooldown_periods

        # Initialize flag for price cut trigger
        self.price_triggered_undefined = False

        # Store slope parameter
        self.slope_threshold = slope_threshold

        # Set initial trend based on the first valid 200-day MA slope
        for i in range(ma_period, len(self.data)):
            slope = self.data['ma_slope'].iloc[i]
            if pd.isna(slope):
                continue  # Skip until we have a valid slope
            if slope > self.slope_threshold:
                initial_trend = 'bull'
            elif slope < -self.slope_threshold:
                initial_trend = 'bear'
            else:
                # If slope is between -slope_threshold and slope_threshold, default to bull
                # Note: 'undefined' state can only be triggered by price cuts below 200MA
                initial_trend = 'bull'
            break  # Found the first valid slope, set the initial trend

        # Set the trend for all rows up to the first valid point
        self.data.loc[self.data.index[:i+1], 'trend'] = initial_trend

        # Initial trend determination
        current_trend = initial_trend
        last_change_idx = i

        for i in range(i + 1, len(self.data)):
            cooldown = self.cooldown_periods[current_trend]
            if i - last_change_idx < cooldown:
                self.data.loc[self.data.index[i], 'trend'] = current_trend
                continue

            slope = self.data['ma_slope'].iloc[i]
            price = self.data['Close'].iloc[i]
            ma_200 = self.data['ma_200'].iloc[i]
            new_trend = current_trend

            # Rule 1: If slope < -slope_threshold, transition to bear market
            if slope < -self.slope_threshold:
                new_trend = 'bear'
                self.price_triggered_undefined = False  # Reset flag when entering bear market
            # Rule 2: If in bull market and price cuts below 200MA, transition to undefined
            # This is the ONLY way to enter the 'undefined' state
            elif current_trend == 'bull' and not pd.isna(ma_200) and price < ma_200:
                new_trend = 'undefined'
                self.price_triggered_undefined = True
            # Rule 3: If slope > slope_threshold, transition to bull
            elif slope > self.slope_threshold:
                if current_trend == 'bear':
                    new_trend = 'bull'
                    self.price_triggered_undefined = False
                elif current_trend == 'undefined':
                    if not self.price_triggered_undefined:
                        new_trend = 'bull'
            # If -slope_threshold <= slope <= slope_threshold, remain in current state
            # No automatic transition to 'undefined' here

            if new_trend != current_trend:
                current_trend = new_trend
                last_change_idx = i

            self.data.loc[self.data.index[i], 'trend'] = current_trend

    def update_trends_after_signals(self):
        """
        Re-evaluate trends, ensuring slope < -slope_threshold triggers bear market
        and price cuts below 200MA trigger undefined state. Track if potential state changes
        are blocked by the cooldown period.
        """
        # Initialize blocked_by_cooldown column
        self.data['blocked_by_cooldown'] = False

        current_trend = self.data['trend'].iloc[0]
        last_change_idx = 0

        for i in range(len(self.data)):
            cooldown = self.cooldown_periods[current_trend]
            slope = self.data['ma_slope'].iloc[i]
            price = self.data['Close'].iloc[i]
            ma_200 = self.data['ma_200'].iloc[i]
            new_trend = current_trend
            transition_reason = ''
            potential_new_trend = current_trend

            # Determine what the new trend would be without cooldown
            # Rule 1: If slope < -slope_threshold, transition to bear market
            if slope < -self.slope_threshold:
                potential_new_trend = 'bear'
            # Rule 2: If in bull market and price cuts below 200MA, transition to undefined
            elif current_trend == 'bull' and not pd.isna(ma_200) and price < ma_200:
                potential_new_trend = 'undefined'
            # Rule 3: If slope > slope_threshold, transition to bull
            elif slope > self.slope_threshold:
                if current_trend == 'bear':
                    potential_new_trend = 'bull'
                elif current_trend == 'undefined':
                    if not self.price_triggered_undefined:
                        potential_new_trend = 'bull'

            # Check if a potential state change is blocked by cooldown
            if i - last_change_idx < cooldown and potential_new_trend != current_trend:
                self.data.loc[self.data.index[i], 'blocked_by_cooldown'] = True

            # Apply cooldown check for actual state change
            if i - last_change_idx < cooldown:
                self.data.loc[self.data.index[i], 'trend'] = current_trend
                continue

            # Apply the actual state change logic
            # Rule 1: If slope < -slope_threshold, transition to bear market
            if slope < -self.slope_threshold:
                new_trend = 'bear'
                transition_reason = f'Relative Slope < -{self.slope_threshold:.4f}'
                self.price_triggered_undefined = False  # Reset flag when entering bear market
            # Rule 2: If in bull market and price cuts below 200MA, transition to undefined
            # This is the ONLY way to enter the 'undefined' state
            elif current_trend == 'bull' and not pd.isna(ma_200) and price < ma_200:
                new_trend = 'undefined'
                transition_reason = 'Price Below 200MA'
                self.price_triggered_undefined = True
            # Rule 3: If slope > slope_threshold, transition to bull
            elif slope > self.slope_threshold:
                if current_trend == 'bear':
                    new_trend = 'bull'
                    transition_reason = f'Relative Slope > {self.slope_threshold:.4f} in Bear Market'
                    self.price_triggered_undefined = False
                elif current_trend == 'undefined':
                    if not self.price_triggered_undefined:
                        new_trend = 'bull'
                        transition_reason = f'Relative Slope > {self.slope_threshold:.4f} in Undefined Market'
            # If -slope_threshold <= slope <= slope_threshold, remain in current state
            # No automatic transition to 'undefined' here

            if new_trend != current_trend:
                current_trend = new_trend
                last_change_idx = i
                self.data.loc[self.data.index[i], 'transition_reason'] = transition_reason

            self.data.loc[self.data.index[i], 'trend'] = current_trend

    def test_window(self, window: int, sd_multiplier: float = 2.0,
                    signal_window: int = 10, min_signals: int = 3) -> dict:
        """
        Test a specific window size with trend separation (no volume signals in this version)

        Parameters:
        window (int): Window size to test (not used in this version)
        sd_multiplier (float): Number of standard deviations for threshold (not used in this version)
        signal_window (int): Window to check for nearby signals (not used in this version)
        min_signals (int): Minimum number of signals required within signal_window (not used in this version)

        Returns:
        dict: Performance metrics by trend
        """
        # Initialize transition_reason column with string dtype
        self.data['transition_reason'] = pd.Series(dtype='string')

        # No volume signals in this version, so no signals to validate
        self.signals = pd.Series(False, index=self.data.index)  # Dummy series for compatibility

        # Re-evaluate trends
        self.update_trends_after_signals()

        # Separate signals by trend (no signals in this version, but kept for compatibility)
        bull_signals = self.signals & (self.data['trend'] == 'bull')
        bear_signals = self.signals & (self.data['trend'] == 'bear')
        undefined_signals = self.signals & (self.data['trend'] == 'undefined')

        # Calculate metrics for each trend
        def calc_metrics(returns, signal_mask):
            ret = returns[signal_mask].dropna()
            total = signal_mask.sum()
            avg_ret = ret.mean() if total > 0 else 0
            vol = ret.std() if total > 5 else 0
            win = (ret > 0).sum() / len(ret) if len(ret) > 0 else 0
            sharpe = (avg_ret / vol * np.sqrt(252)) if vol != 0 else 0
            return total, avg_ret, vol, win, sharpe

        # 10-period metrics
        bull_10 = calc_metrics(self.data['return_10'], bull_signals)
        bear_10 = calc_metrics(self.data['return_10'], bear_signals)
        undefined_10 = calc_metrics(self.data['return_10'], undefined_signals)

        # 20-period metrics
        bull_20 = calc_metrics(self.data['return_20'], bull_signals)
        bear_20 = calc_metrics(self.data['return_20'], bear_signals)
        undefined_20 = calc_metrics(self.data['return_20'], undefined_signals)

        return {
            'window': window,
            'total_signals': self.signals.sum(),
            'bull_signals': bull_10[0], 'bull_avg_return_10': bull_10[1],
            'bull_volatility_10': bull_10[2], 'bull_win_rate_10': bull_10[3],
            'bull_sharpe_10': bull_10[4], 'bull_avg_return_20': bull_20[1],
            'bull_volatility_20': bull_20[2], 'bull_win_rate_20': bull_20[3],
            'bull_sharpe_20': bull_20[4],
            'bear_signals': bear_10[0], 'bear_avg_return_10': bear_10[1],
            'bear_volatility_10': bear_10[2], 'bear_win_rate_10': bear_10[3],
            'bear_sharpe_10': bear_10[4], 'bear_avg_return_20': bear_20[1],
            'bear_volatility_20': bear_20[2], 'bear_win_rate_20': bear_20[3],
            'bear_sharpe_20': bear_20[4],
            'undefined_signals': undefined_10[0], 'undefined_avg_return_10': undefined_10[1],
            'undefined_volatility_10': undefined_10[2], 'undefined_win_rate_10': undefined_10[3],
            'undefined_sharpe_10': undefined_10[4], 'undefined_avg_return_20': undefined_20[1],
            'undefined_volatility_20': undefined_20[2], 'undefined_win_rate_20': undefined_20[3],
            'undefined_sharpe_20': undefined_20[4]
        }

    def optimize_window(self, window_range: List[int], sd_multiplier: float = 3.0,
                        signal_window: int = 10, min_signals: int = 3,
                        slope_threshold: float = 0.001) -> pd.DataFrame:
        """
        Test multiple window sizes and return results (simplified for this version)

        Parameters:
        window_range (List[int]): List of window sizes to test
        sd_multiplier (float): Number of standard deviations for threshold (not used in this version)
        signal_window (int): Window to check for nearby signals (not used in this version)
        min_signals (int): Minimum number of signals required within signal_window (not used in this version)
        slope_threshold (float): Threshold for relative MA slope to determine trend (default: 0.001)

        Returns:
        pd.DataFrame: Results for all tested windows
        """
        results = []
        for window in window_range:
            if window < len(self.data) - 20:  # Ensure enough data for 20-period returns
                result = self.test_window(window, sd_multiplier, signal_window, min_signals)
                results.append(result)

        return pd.DataFrame(results)

    def plot_trends(self, log_scale: bool = False):
        """
        Plot Close price with colored segments for each trend state and vertical lines at state changes.
        Allows choosing between logarithmic or regular price scale.
        Reflects cooldown periods: bull=0, undefined=30.

        Parameters:
        log_scale (bool): If True, plot the price on a logarithmic scale; if False, use a regular (linear) scale (default: False)
        """
        plt.figure(figsize=(12, 6), facecolor='white')

        # Plot Close price
        plt.plot(self.data.index, self.data['Close'], label='Close', color='black', linewidth=1)

        # Plot 200-day MA
        plt.plot(self.data.index, self.data['ma_200'], label='200-day MA', color='blue', linestyle='--', linewidth=1)

        # Set y-axis scale based on log_scale parameter
        if log_scale:
            plt.yscale('log')
            # Define the band as multiplicative factors for log scale (±5% of MA)
            lower_band = self.data['ma_200'] * 0.95  # 5% below MA
            upper_band = self.data['ma_200'] * 1.05  # 5% above MA
            y_label = 'Log Price'
            title_prefix = 'Log Close Price'
        else:
            # Define the band as a linear offset for regular scale (±5% of MA)
            ma_band = self.data['ma_200'] * 0.05
            lower_band = self.data['ma_200'] - ma_band
            upper_band = self.data['ma_200'] + ma_band
            y_label = 'Price'
            title_prefix = 'Close Price'

        # Color segments for each trend, filling around the MA
        trends = self.data['trend']
        colors = {'bull': 'green', 'bear': 'red', 'undefined': 'orange'}

        for trend in ['bull', 'bear', 'undefined']:
            mask = (trends == trend)
            plt.fill_between(self.data.index,
                             lower_band,
                             upper_band,
                             where=mask,
                             color=colors[trend],
                             alpha=0.2,
                             label=trend.capitalize())

        # Draw vertical lines at state changes
        trends = self.data['trend']
        for i in range(1, len(trends)):
            if trends.iloc[i] != trends.iloc[i - 1]:
                # State change detected, draw a vertical line
                plt.axvline(x=self.data.index[i], color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

        # Customize plot appearance
        plt.title(
            f'{title_prefix} with Trend States (Price Below 200MA Triggers Undefined, Cooldown: Bull=0, Undefined=30)',
            fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend(loc='upper left')
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        plt.show()

    def export_market_conditions_to_csv(self, filename: str = "market_conditions.csv"):
        """
        Export market conditions and their time periods to a CSV file, including the reason for each transition,
        the duration of each state period, and whether a potential state change was blocked by the cooldown period.
        Reflects cooldown periods: bull=0, undefined=30.

        Parameters:
        filename (str): Name of the CSV file to save (default: 'market_conditions.csv')

        Returns:
        pd.DataFrame: DataFrame containing market conditions, their time periods, transition reasons, durations,
        and whether a potential state change was blocked by cooldown
        """
        # Ensure the index is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                raise ValueError(f"Could not convert DataFrame index to datetime format: {e}")

        # Initialize lists to store market conditions, their periods, transition reasons, durations, and blocked status
        conditions = []
        start_dates = []
        end_dates = []
        transition_reasons = []
        durations = []
        blocked_by_cooldown_list = []

        # Get the trend series
        trends = self.data['trend']

        # Initialize variables for tracking periods
        current_condition = trends.iloc[0]
        start_idx = 0
        # The reason for the first period is not applicable
        current_reason = 'Initial State'

        # Iterate through the trends to find consecutive periods
        for i in range(1, len(trends)):
            if trends.iloc[i] != current_condition:
                # End of a period
                conditions.append(current_condition)
                start_date = self.data.index[start_idx]
                end_date = self.data.index[i - 1]
                start_dates.append(start_date.strftime('%Y-%m-%d'))
                end_dates.append(end_date.strftime('%Y-%m-%d'))
                transition_reasons.append(current_reason)
                # Calculate duration in days
                duration = (end_date - start_date).days + 1  # +1 to include both start and end dates
                durations.append(duration)

                # Check if a potential state change was blocked by cooldown during this period
                period_data = self.data.iloc[start_idx:i]
                blocked = period_data['blocked_by_cooldown'].any()
                blocked_by_cooldown_list.append(blocked)

                # Start of a new period
                current_condition = trends.iloc[i]
                start_idx = i
                # The reason for the new period is the reason for the transition at index i
                current_reason = self.data['transition_reason'].iloc[i] if self.data['transition_reason'].iloc[
                    i] else 'Unknown'

        # Add the last period
        conditions.append(current_condition)
        start_date = self.data.index[start_idx]
        end_date = self.data.index[-1]
        start_dates.append(start_date.strftime('%Y-%m-%d'))
        end_dates.append(end_date.strftime('%Y-%m-%d'))
        transition_reasons.append(current_reason)
        # Calculate duration in days for the last period
        duration = (end_date - start_date).days + 1  # +1 to include both start and end dates
        durations.append(duration)

        # Check if a potential state change was blocked by cooldown during the last period
        period_data = self.data.iloc[start_idx:]
        blocked = period_data['blocked_by_cooldown'].any()
        blocked_by_cooldown_list.append(blocked)

        # Create a DataFrame
        market_conditions_df = pd.DataFrame({
            'Market Condition': conditions,
            'Start Date': start_dates,
            'End Date': end_dates,
            'Duration (Days)': durations,
            'Transition Reason': transition_reasons,
            'Blocked by Cooldown': blocked_by_cooldown_list
        })

        # Save to CSV
        market_conditions_df.to_csv(filename, index=False)
        print(f"Market conditions exported to {filename}")

        return market_conditions_df

if __name__ == "__main__":
    start_date = SPX_datadownload.INDEX_CLOSE_FILENAME_TRAIN # fixed
    end_date = SPX_datadownload.INDEX_VOLUME_FILENAME_TRAIN

    spx_close = pd.read_csv(start_date, index_col='Date', header=0)
    spx_vol = pd.read_csv(end_date, index_col='Date', header=0)
    size = 7000  # approx 250 = 1y
    spx_closevol = pd.concat([spx_close, spx_vol], axis=1) #.tail(size)
    print(spx_closevol)

    # Create backtester
    backtester = VolumeIndicatorBacktest(spx_closevol, slope_threshold=0.0002)

    # Test window sizes from a to b in steps of c
    window_range = range(200, 201, 50)
    results = backtester.optimize_window(window_range)

    print("Optimization Results:")
    print(results)

    # Export market conditions to CSV
    market_conditions_df = backtester.export_market_conditions_to_csv("market_conditions_output_v5.1_onlyMA_removebulladdvola_??.csv")
    print("Market Conditions:")
    print(market_conditions_df)

    # Plot trends
    backtester.plot_trends()
    backtester.plot_trends(log_scale=True)

    # # Inspect the typical range of self.data['ma_slope']
    # print(backtester.data['ma_slope'].describe())

