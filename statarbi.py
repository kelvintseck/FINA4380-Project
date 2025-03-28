import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Step 1: Download historical price data for the pair
def get_data(ticker1, ticker2, start_date, end_date):
    stock1 = yf.download(ticker1, start=start_date, end=end_date)['Close']
    stock2 = yf.download(ticker2, start=start_date, end=end_date)['Close']
    data = pd.concat([stock1, stock2], axis=1).dropna()
    data.columns = [ticker1, ticker2]
    return data


# Step 2: Calculate the hedge ratio (β) and spread
def calculate_spread(data, lookback_period):
    # Use a rolling window to estimate the hedge ratio
    X = sm.add_constant(data.iloc[:, 1])  # Stock 2 as independent variable
    y = data.iloc[:, 0]  # Stock 1 as dependent variable
    model = sm.OLS(y, X).fit()
    beta = model.params[1]  # Hedge ratio

    # Calculate spread: Stock1 - β * Stock2
    spread = data.iloc[:, 0] - beta * data.iloc[:, 1]
    return spread, beta


# Step 3: Generate trading signals based on mean reversion
def generate_signals(spread, window=20):
    # Calculate rolling mean and standard deviation of the spread
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()

    # Z-score: How many standard deviations away from the mean
    z_score = (spread - spread_mean) / spread_std

    # Define entry/exit thresholds
    entry_threshold = 2.0  # 2 standard deviations
    exit_threshold = 0.5  # Exit near the mean

    # Initialize signals
    signals = pd.Series(0, index=spread.index)

    # Long the spread (buy Stock1, sell Stock2) when undervalued
    signals[z_score <= -entry_threshold] = 1
    # Short the spread (sell Stock1, buy Stock2) when overvalued
    signals[z_score >= entry_threshold] = -1
    # Exit positions when spread nears the mean
    signals[(z_score.abs() <= exit_threshold) & (signals.shift(1) != 0)] = 0

    # Forward fill to hold positions until exit
    signals = signals.ffill()
    return signals, z_score


# Step 4: Simulate the strategy
def simulate_strategy(data, signals, beta):
    # Position in Stock1: 1 unit when long, -1 when short
    stock1_position = signals

    # Position in Stock2: Opposite of Stock1, adjusted by hedge ratio
    stock2_position = -signals * beta

    # Calculate daily returns
    daily_returns_stock1 = data.iloc[:, 0].pct_change()
    daily_returns_stock2 = data.iloc[:, 1].pct_change()

    # Portfolio returns: Stock1 returns + Stock2 returns
    portfolio_returns = (stock1_position.shift(1) * daily_returns_stock1 +
                         stock2_position.shift(1) * daily_returns_stock2)

    # Cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns, portfolio_returns


# Step 5: Visualize results
def plot_results(data, spread, z_score, signals, cumulative_returns):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot prices
    ax1.plot(data.iloc[:, 0], label=data.columns[0])
    ax1.plot(data.iloc[:, 1], label=data.columns[1])
    ax1.set_title("Stock Prices")
    ax1.legend()

    # Plot spread and Z-score
    ax2.plot(spread, label="Spread")
    ax2.plot(spread.rolling(20).mean(), label="20-day Mean")
    ax2.fill_between(spread.index, spread.rolling(20).mean() - 2 * spread.rolling(20).std(),
                     spread.rolling(20).mean() + 2 * spread.rolling(20).std(), alpha=0.2)
    ax2.set_title("Spread")
    ax2.legend()

    # Plot cumulative returns
    ax3.plot(cumulative_returns, label="Cumulative Returns")
    ax3.set_title("Strategy Performance")
    ax3.legend()

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Parameters
    ticker1, ticker2 = "KO", "PEP"  # Coca-Cola and PepsiCo
    start_date = "2022-01-01"
    end_date = "2025-03-28"  # Up to current date (March 28, 2025)
    lookback_period = 252  # 1 year lookback for beta
    window = 20  # Rolling window for mean and std

    # Get data
    data = get_data(ticker1, ticker2, start_date, end_date)

    # Calculate spread and hedge ratio
    spread, beta = calculate_spread(data, lookback_period)

    # Generate trading signals
    signals, z_score = generate_signals(spread, window)

    # Simulate strategy
    cumulative_returns, portfolio_returns = simulate_strategy(data, signals, beta)

    # Plot results
    plot_results(data, spread, z_score, signals, cumulative_returns)

    # Print summary
    print(f"Hedge Ratio (β): {beta:.2f}")
    print(f"Total Return: {(cumulative_returns[-1] - 1) * 100:.2f}%")
    print(f"Sharpe Ratio: {(portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252):.2f}")