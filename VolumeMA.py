import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import SPX_datadownload

''' 
rationale: to see if spike in volume -> volatility in the future / change of state
1. Run SPX_datadownload.py in prior
'''

""" Problem
feature "volume_triggered_undefined" seems to have a problem
"""

# may add: volumeSignalCD to minimize noise / false signals
# add 200MA direction to seperate two sizes, with condition that even if in uptrend, when price break below 200MA, it becomes undefined,
# same case when in bear market, if price break above 200MA, it becomes undefined.
# until 200MA direction change, so that bull -> undefined -> bear -> undefined


class VolumeIndicatorBacktest:
    def __init__(self, data: pd.DataFrame, ma_period: int = 200,
                 cooldown_periods: dict = None):
        """
        Initialize with historical data and calculate market trend

        Parameters:
        data (pd.DataFrame): DataFrame with 'Volume' and 'Close' columns
        ma_period (int): Period for moving average to determine trend (default: 200)
        cooldown_periods (dict): Cooldown periods for each trend {'bull': int, 'bear': int, 'undefined': int}
        """
        self.data = data.copy()
        # Calculate future returns
        self.data['return_10'] = self.data['Close'].pct_change(periods=10).shift(-10)
        self.data['return_20'] = self.data['Close'].pct_change(periods=20).shift(-20)

        # Calculate 200-day MA and its slope
        self.data['ma_200'] = self.data['Close'].rolling(window=ma_period).mean()
        self.data['ma_slope'] = self.data['ma_200'].diff()

        # Initialize trend column
        self.data['trend'] = 'undefined'

        # Set cooldown periods for each trend
        if cooldown_periods is None:
            cooldown_periods = {'bull': 130, 'bear': 0, 'undefined': 0}
        self.cooldown_periods = cooldown_periods

        # Initialize flag for volume signal trigger
        self.volume_triggered_undefined = False

        # Determine trend with transitions and state-specific cooldown
        current_trend = 'undefined'
        last_change_idx = 0

        for i in range(ma_period, len(self.data)):
            cooldown = self.cooldown_periods[current_trend]
            if i - last_change_idx < cooldown:
                self.data.loc[self.data.index[i], 'trend'] = current_trend
                continue

            slope = self.data['ma_slope'].iloc[i]

            new_trend = current_trend

            # Rule 1: If slope < 0, always transition to bear market
            if slope < 0:
                new_trend = 'bear'
                self.volume_triggered_undefined = False
            else:
                # Rule 2: Bull to undefined only via volume signal (handled in test_window)
                if current_trend == 'bear':
                    if slope >= 0:  # Bear market ends when MA slope turns positive
                        new_trend = 'undefined'
                        self.volume_triggered_undefined = False
                elif current_trend == 'undefined':
                    if slope >= 0 and not self.volume_triggered_undefined:
                        new_trend = 'bull'

            if new_trend != current_trend:
                current_trend = new_trend
                last_change_idx = i

            self.data.loc[self.data.index[i], 'trend'] = current_trend

    def test_window(self, window: int, sd_multiplier: float = 2.0,
                    signal_window: int = 10, min_signals: int = 3) -> dict:
        """
        Test a specific window size with trend separation and validate volume signals

        Parameters:
        window (int): Window size to test
        sd_multiplier (float): Number of standard deviations for threshold
        signal_window (int): Window to check for nearby signals (default: 10)
        min_signals (int): Minimum number of signals required within signal_window (default: 2)

        Returns:
        dict: Performance metrics by trend
        """
        signals = []
        volume_series = pd.Series(dtype=float)

        for volume in self.data['Volume']:
            volume_series = pd.concat([volume_series, pd.Series([volume])],
                                      ignore_index=True)
            if len(volume_series) >= window:
                rolling_mean = volume_series.rolling(window=window).mean().iloc[-1]
                rolling_std = volume_series.rolling(window=window).std().iloc[-1]
                threshold = rolling_mean + (sd_multiplier * rolling_std)
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
                    # Update trend to undefined from this point until bear market
                    self.data.loc[self.data.index[i:], 'trend'] = 'undefined'
                    self.volume_triggered_undefined = True

        self.signals = validated_signals  # Store validated signals for plotting

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

    def optimize_window(self, window_range: List[int], sd_multiplier: float = 3.0) -> pd.DataFrame:
        """
        Test multiple window sizes and return results

        Parameters:
        window_range (List[int]): List of window sizes to test
        sd_multiplier (float): Number of standard deviations for threshold

        Returns:
        pd.DataFrame: Results for all tested windows
        """
        results = []
        for window in window_range:
            if window < len(self.data) - 20:  # Ensure enough data for 20-period returns
                result = self.test_window(window, sd_multiplier)
                results.append(result)

        return pd.DataFrame(results)

    def plot_trends(self):
        """
        Plot Close price with colored segments for each trend state and volume signals (bull and undefined markets)
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
        Export market conditions and their time periods to a CSV file.

        Parameters:
        filename (str): Name of the CSV file to save (default: 'market_conditions.csv')

        Returns:
        pd.DataFrame: DataFrame containing market conditions and their time periods
        """
        # Ensure the index is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                raise ValueError(f"Could not convert DataFrame index to datetime format: {e}")

        # Initialize lists to store market conditions and their periods
        conditions = []
        start_dates = []
        end_dates = []

        # Get the trend series
        trends = self.data['trend']

        # Initialize variables for tracking periods
        current_condition = trends.iloc[0]
        start_idx = 0

        # Iterate through the trends to find consecutive periods
        for i in range(1, len(trends)):
            if trends.iloc[i] != current_condition:
                # End of a period
                conditions.append(current_condition)
                start_dates.append(self.data.index[start_idx].strftime('%Y-%m-%d'))
                end_dates.append(self.data.index[i - 1].strftime('%Y-%m-%d'))

                # Start of a new period
                current_condition = trends.iloc[i]
                start_idx = i

        # Add the last period
        conditions.append(current_condition)
        start_dates.append(self.data.index[start_idx].strftime('%Y-%m-%d'))
        end_dates.append(self.data.index[-1].strftime('%Y-%m-%d'))

        # Create a DataFrame
        market_conditions_df = pd.DataFrame({
            'Market Condition': conditions,
            'Start Date': start_dates,
            'End Date': end_dates
        })

        # Save to CSV
        market_conditions_df.to_csv(filename, index=False)
        print(f"Market conditions exported to {filename}")

        return market_conditions_df

if __name__ == "__main__":
    spx_close = pd.read_csv(SPX_datadownload.INDEX_CLOSE_FILENAME, index_col='Date', header=0)
    spx_vol = pd.read_csv(SPX_datadownload.INDEX_VOLUME_FILENAME, index_col='Date', header=0)
    size = 5000 # approx 250 = 1y
    spx_closevol = pd.concat([spx_close, spx_vol], axis=1).tail(size)
    print(spx_closevol)

    # Create backtester
    backtester = VolumeIndicatorBacktest(spx_closevol)

    # Test window sizes from a to b in steps of c
    window_range = range(200, 201, 50)
    results = backtester.optimize_window(window_range)

    # Print results (subset of columns)
    print(results[['window', 'total_signals',
                   'bull_signals', 'bull_sharpe_10', 'bull_sharpe_20',
                   'bear_signals', 'bear_sharpe_10', 'bear_sharpe_20',
                   'undefined_signals', 'undefined_sharpe_10', 'undefined_sharpe_20']].round(3))

    # Find best window
    print("\nBest window based on bull market 20-period Sharpe:")
    best_bull = results.loc[results['bull_sharpe_20'].idxmax()]
    print(best_bull[['window', 'bull_signals', 'bull_sharpe_20',
                     'bear_sharpe_20', 'undefined_sharpe_20']].round(3))

    optimal_window = int(best_bull['window'])
    print(f"\nRecommended window size: {optimal_window}")

    # Export market conditions to CSV
    market_conditions_df = backtester.export_market_conditions_to_csv("market_conditions_output.csv")

    # Optionally, print the DataFrame to see the results
    print(market_conditions_df)

    # Plot the trends
    backtester.plot_trends()
