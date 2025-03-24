import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import SPX_datadownload

"""
diff from v3:
slope scalar issue solved
"""

class VolumeIndicatorBacktest:
    def __init__(self, data: pd.DataFrame, ma_period: int = 200,
                 cooldown_periods: dict = None, slope_threshold: float = 0.001):
        """
        Initialize with historical data and calculate market trend

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

        # Initialize trend column as NaN (will be set based on slope)
        self.data['trend'] = np.nan

        # Set cooldown periods for each trend
        if cooldown_periods is None:
            cooldown_periods = {'bull': 130, 'bear': 0, 'undefined': 0}
        self.cooldown_periods = cooldown_periods

        # Initialize flag for volume signal trigger
        self.volume_triggered_undefined = False

        # Store slope threshold
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
                initial_trend = 'undefined'
            break  # Found the first valid slope, set the initial trend

        # Set the trend for all rows up to the first valid point
        self.data.loc[self.data.index[:i + 1], 'trend'] = initial_trend

        # Initial trend determination (before volume signals)
        current_trend = initial_trend
        last_change_idx = i

        for i in range(i + 1, len(self.data)):
            cooldown = self.cooldown_periods[current_trend]
            if i - last_change_idx < cooldown:
                self.data.loc[self.data.index[i], 'trend'] = current_trend
                continue

            slope = self.data['ma_slope'].iloc[i]
            new_trend = current_trend

            # Rule 1: If slope < -slope_threshold, transition to bear market
            if slope < -self.slope_threshold:
                new_trend = 'bear'
                self.volume_triggered_undefined = False
            # Rule 2: If slope > slope_threshold, allow transitions to bull or undefined
            elif slope > self.slope_threshold:
                if current_trend == 'bear':
                    new_trend = 'undefined'
                    self.volume_triggered_undefined = False
                elif current_trend == 'undefined':
                    if not self.volume_triggered_undefined:
                        new_trend = 'bull'
            # If -slope_threshold <= slope <= slope_threshold, remain in current state

            if new_trend != current_trend:
                current_trend = new_trend
                last_change_idx = i

            self.data.loc[self.data.index[i], 'trend'] = current_trend

    def is_witching_day(self, date: pd.Timestamp) -> bool:
        """
        Check if a given date is a witching day (third Friday of March, June, September, December).

        Parameters:
        date (pd.Timestamp): Date to check

        Returns:
        bool: True if the date is a witching day, False otherwise
        """
        # Witching days occur in March (3), June (6), September (9), and December (12)
        if date.month not in [3, 6, 9, 12]:
            return False

        # Check if the date is a Friday (weekday 4, where Monday is 0)
        if date.weekday() != 4:
            return False

        # Check if the date is the third Friday of the month
        # The third Friday will be between the 15th and 21st of the month
        return 15 <= date.day <= 21

    def update_trends_after_signals(self):
        """
        Re-evaluate trends after volume signals, ensuring slope < -slope_threshold triggers bear market.
        """
        current_trend = self.data['trend'].iloc[0]
        last_change_idx = 0

        for i in range(len(self.data)):
            cooldown = self.cooldown_periods[current_trend]
            if i - last_change_idx < cooldown:
                self.data.loc[self.data.index[i], 'trend'] = current_trend
                continue

            slope = self.data['ma_slope'].iloc[i]
            new_trend = current_trend
            transition_reason = ''

            # Rule 1: If slope < -slope_threshold, transition to bear market
            if slope < -self.slope_threshold:
                new_trend = 'bear'
                transition_reason = f'Relative Slope < -{self.slope_threshold:.4f}'
                self.volume_triggered_undefined = False
            # Rule 2: If slope > slope_threshold, allow transitions to bull or undefined
            elif slope > self.slope_threshold:
                if current_trend == 'bear':
                    new_trend = 'undefined'
                    transition_reason = f'Relative Slope > {self.slope_threshold:.4f} in Bear Market'
                    self.volume_triggered_undefined = False
                elif current_trend == 'undefined':
                    if not self.volume_triggered_undefined:
                        new_trend = 'bull'
                        transition_reason = f'Relative Slope > {self.slope_threshold:.4f} in Undefined Market'
            # If -slope_threshold <= slope <= slope_threshold, remain in current state

            if new_trend != current_trend:
                current_trend = new_trend
                last_change_idx = i
                self.data.loc[self.data.index[i], 'transition_reason'] = transition_reason

            self.data.loc[self.data.index[i], 'trend'] = current_trend

    def test_window(self, window: int, sd_multiplier: float = 2.0,
                    signal_window: int = 10, min_signals: int = 3) -> dict:
        """
        Test a specific window size with trend separation and validate volume signals

        Parameters:
        window (int): Window size to test
        sd_multiplier (float): Number of standard deviations for threshold
        signal_window (int): Window to check for nearby signals (default: 10)
        min_signals (int): Minimum number of signals required within signal_window (default: 3)

        Returns:
        dict: Performance metrics by trend
        """
        signals = []
        volume_series = pd.Series(dtype=float)

        # Initialize transition_reason column
        self.data['transition_reason'] = ''

        for i, (volume, date) in enumerate(zip(self.data['Volume'], self.data.index)):
            volume_series = pd.concat([volume_series, pd.Series([volume])],
                                      ignore_index=True)
            if len(volume_series) >= window:
                rolling_mean = volume_series.rolling(window=window).mean().iloc[-1]
                rolling_std = volume_series.rolling(window=window).std().iloc[-1]
                threshold = rolling_mean + (sd_multiplier * rolling_std)
                # Check if the date is a witching day
                if self.is_witching_day(date):
                    signals.append(False)  # Exclude witching days from volume signals
                else:
                    signals.append(volume > threshold)
            else:
                signals.append(False)

        raw_signals = pd.Series(signals, index=self.data.index)

        # Validate signals: require at least min_signals within signal_window periods
        validated_signals = raw_signals.copy()
        for i in range(len(raw_signals)):
            if raw_signals.iloc[i]:
                start_idx = max(0, i - signal_window)
                end_idx = min(len(raw_signals), i + signal_window + 1)
                nearby_signals = raw_signals.iloc[start_idx:end_idx].sum()
                if nearby_signals < min_signals:
                    validated_signals.iloc[i] = False
                # If signal is valid and in bull market, trigger undefined state
                elif (self.data['trend'].iloc[i] == 'bull'):
                    # Update trend to undefined from this point
                    self.data.loc[self.data.index[i:], 'trend'] = 'undefined'
                    self.data.loc[self.data.index[i], 'transition_reason'] = 'Volume Signal in Bull Market'
                    self.volume_triggered_undefined = True

        self.signals = validated_signals  # Store validated signals for plotting

        # Re-evaluate trends to ensure slope < -slope_threshold triggers bear market
        self.update_trends_after_signals()

        # Separate signals by trend
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
        Test multiple window sizes and return results

        Parameters:
        window_range (List[int]): List of window sizes to test
        sd_multiplier (float): Number of standard deviations for threshold
        signal_window (int): Window to check for nearby signals (default: 10)
        min_signals (int): Minimum number of signals required within signal_window (default: 3)
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

    def plot_trends(self):
        """
        Plot Close price with colored segments for each trend state, volume signals (bull and undefined markets),
        and vertical lines at state changes
        """
        plt.figure(figsize=(12, 6), facecolor='white')

        # Plot Close price
        plt.plot(self.data.index, self.data['Close'], label='Close', color='black', linewidth=1)

        # Plot 200-day MA
        plt.plot(self.data.index, self.data['ma_200'], label='200-day MA', color='blue', linestyle='--', linewidth=1)

        # Color segments for each trend, filling around the MA
        trends = self.data['trend']
        colors = {'bull': 'green', 'bear': 'red', 'undefined': 'orange'}

        ma_band = self.data['ma_200'] * 0.05  # 5% of MA for the band width

        for trend in ['bull', 'bear', 'undefined']:
            mask = (trends == trend)
            plt.fill_between(self.data.index,
                             self.data['ma_200'] - ma_band,
                             self.data['ma_200'] + ma_band,
                             where=mask,
                             color=colors[trend],
                             alpha=0.2,
                             label=trend.capitalize())

        # Plot volume signals for bull and undefined markets
        if hasattr(self, 'signals'):
            # Bull market signals
            bull_signals = self.data[self.signals & (self.data['trend'] == 'bull')]
            plt.scatter(bull_signals.index, bull_signals['Close'],
                        marker='^', s=100, color='green',
                        label='Bull Volume Signal', zorder=5)

            # Undefined market signals
            undefined_signals = self.data[self.signals & (self.data['trend'] == 'undefined')]
            plt.scatter(undefined_signals.index, undefined_signals['Close'],
                        marker='^', s=100, color='orange',
                        label='Undefined Volume Signal', zorder=5)

        # Draw vertical lines at state changes
        trends = self.data['trend']
        for i in range(1, len(trends)):
            if trends.iloc[i] != trends.iloc[i - 1]:
                # State change detected, draw a vertical line
                plt.axvline(x=self.data.index[i], color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

        # Customize plot appearance
        plt.title('Close Price with Trend States and Volume Signals (Bull & Undefined)', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(loc='upper left')
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        plt.show()

    def export_market_conditions_to_csv(self, filename: str = "market_conditions.csv"):
        """
        Export market conditions and their time periods to a CSV file, including the reason for each transition.

        Parameters:
        filename (str): Name of the CSV file to save (default: 'market_conditions.csv')

        Returns:
        pd.DataFrame: DataFrame containing market conditions, their time periods, and transition reasons
        """
        # Ensure the index is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                raise ValueError(f"Could not convert DataFrame index to datetime format: {e}")

        # Initialize lists to store market conditions, their periods, and transition reasons
        conditions = []
        start_dates = []
        end_dates = []
        transition_reasons = []

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
                start_dates.append(self.data.index[start_idx].strftime('%Y-%m-%d'))
                end_dates.append(self.data.index[i - 1].strftime('%Y-%m-%d'))
                transition_reasons.append(current_reason)

                # Start of a new period
                current_condition = trends.iloc[i]
                start_idx = i
                # The reason for the new period is the reason for the transition at index i
                current_reason = self.data['transition_reason'].iloc[i] if self.data['transition_reason'].iloc[
                    i] else 'Unknown'

        # Add the last period
        conditions.append(current_condition)
        start_dates.append(self.data.index[start_idx].strftime('%Y-%m-%d'))
        end_dates.append(self.data.index[-1].strftime('%Y-%m-%d'))
        transition_reasons.append(current_reason)

        # Create a DataFrame
        market_conditions_df = pd.DataFrame({
            'Market Condition': conditions,
            'Start Date': start_dates,
            'End Date': end_dates,
            'Transition Reason': transition_reasons
        })

        # Save to CSV
        market_conditions_df.to_csv(filename, index=False)
        print(f"Market conditions exported to {filename}")

        return market_conditions_df


if __name__ == "__main__":
    spx_close = pd.read_csv(SPX_datadownload.INDEX_CLOSE_FILENAME, index_col='Date', header=0)
    spx_vol = pd.read_csv(SPX_datadownload.INDEX_VOLUME_FILENAME, index_col='Date', header=0)
    size = 5000 # approx 250 = 1y
    spx_closevol = pd.concat([spx_close, spx_vol], axis=1).tail(size) #.tail(size) #  #
    print(spx_closevol)

    # Create backtester
    backtester = VolumeIndicatorBacktest(spx_closevol, slope_threshold=0.0002)

    # Test window sizes from a to b in steps of c
    window_range = range(200, 201, 50)
    results = backtester.optimize_window(window_range)

    print("Optimization Results:")
    print(results)

    # Step 4: Export market conditions to CSV
    market_conditions_df = backtester.export_market_conditions_to_csv("market_conditions_output_v4_tail5000.csv")
    print("Market Conditions:")
    print(market_conditions_df)

    # Step 5: Plot trends
    backtester.plot_trends()

    ## inspect the typical range of self.data['ma_slope'], to determine an appropriate slope_threshold
    # print(backtester.data['ma_slope'][200:210])
    # print(backtester.data['ma_slope'].tail(10))
    print(backtester.data['ma_slope'].describe())

    # either change slope formula, or take log,