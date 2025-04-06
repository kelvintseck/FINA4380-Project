import pandas as pd
import numpy as np
import os
from scipy import optimize
import datetime
from boardMarketIndex import BoardMarketIndex
import matplotlib.pyplot as plt
from numba import jit
import warnings

# Constants
ROLLING_WINDOW = 360
MAX_WEIGHT = 0.4
MIN_WEIGHT = 0
CASH_OUT_THRESHOLD = 0.01
TOLERANCE = 1e-30
MAX_ITERATIONS = 100
VARIANT = False

def load_data(file_dir):
    """Load all required data files."""
    files = {
        'prices': 'ETFs_daily_prices.csv',
        'avg_returns': 'average_momentum_returns.csv',
        'returns_90d': 'returns_90d.csv',
        'risk_free': 'riskFree.csv'
    }
    warnings.filterwarnings('ignore', category=UserWarning)
    data = {key: pd.read_csv(os.path.join(file_dir, filename), index_col=0, parse_dates=True, dayfirst=True) 
            for key, filename in files.items()}
    # Preprocess data
    data['daily_returns'] = data['prices'].pct_change().dropna()
    return data

def get_top_etfs(etf_list, avg_returns, momentum_flags):
    """Sort ETFs by average returns and filter positive ones."""
    filtered_returns = avg_returns * momentum_flags
    sorted_indices = filtered_returns.argsort()[::-1]
    sorted_returns = filtered_returns.iloc[sorted_indices]
    
    # Find first zero or negative return
    cutoff = (sorted_returns > 0).sum()
    return etf_list[sorted_indices][:cutoff]

@jit(nopython=True)
def calc_negative_sharpe(weights, means, cov_matrix, avg_returns, risk_free, VARIANT):
    """Calculate negative Sharpe Ratio for optimization."""
    if VARIANT == True:
        portfolio_return = np.dot(means, weights)
    else:
        portfolio_return = np.dot(avg_returns, weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(portfolio_return - risk_free) / portfolio_volatility

def optimize_weights(df, avg_returns, risk_free):
    """Optimize portfolio weights using Smart Beta strategy."""
    means = df.mean().to_numpy()
    cov_matrix = df.cov().to_numpy()
    num_assets = len(df.columns)
    
    initial_weights = np.ones(num_assets) / num_assets
    bndsa = [(MIN_WEIGHT, MAX_WEIGHT)] * num_assets
    cons = {'type': 'eq', 'fun': lambda w: 1 - np.sum(w)}
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    result = optimize.minimize(
        calc_negative_sharpe,
        initial_weights,
        args=(means, cov_matrix, avg_returns.to_numpy(), risk_free, VARIANT),
        method='SLSQP',
        bounds=bndsa,
        constraints=cons,
        tol=TOLERANCE,
        options={'maxiter': MAX_ITERATIONS}
    )
    return result.x

def get_rolling_data(df, current_idx, window=ROLLING_WINDOW):
    """Extract rolling window of data."""
    start_idx = max(0, current_idx - window)
    return df.iloc[start_idx:current_idx]

def calculate_portfolio_weights():
    """Main function to calculate and save portfolio weights."""
    file_dir = os.path.dirname(__file__)
    data = load_data(file_dir)
    etf_list = np.array(data['avg_returns'].columns)
    weights_df = pd.DataFrame(0.0, columns=list(data['prices'].columns) + ['cash'], 
                            index=data['prices'].index)
    runtime_df = pd.DataFrame(columns=['Time'], index=data['prices'].index)
    
    for idx, date in enumerate(data['prices'].iloc[:-1826].index):
        start_time = datetime.datetime.now()
        # Skip if no valid average returns
        current_avg_returns = data['avg_returns'].loc[date].fillna(0)
        if current_avg_returns.sum() == 0:
            continue
        risk_free_rate = data['risk_free'].loc[date].iloc[0]
        market_index = BoardMarketIndex(os.path.join(file_dir, "ETFs_daily_prices.csv"), os.path.join(file_dir, "average_momentum_returns.csv"), date.strftime('%Y-%m-%d'))
        
        # Identify ETFs with strong momentum
        momentum_flags = np.array([market_index.asset_is_in_strong_momentum_group(etf) 
                                 for etf in etf_list])
        
        # Get top performing ETFs
        top_etfs = get_top_etfs(etf_list, current_avg_returns, momentum_flags)
        rolling_returns = get_rolling_data(data['daily_returns'][top_etfs], idx)
        
        # Optimize weights
        optimal_weights = optimize_weights(rolling_returns, current_avg_returns[top_etfs], risk_free_rate)
        
        # Handle small weights and cash allocation
        cash_weight = 0.0
        for i, weight in enumerate(optimal_weights):
            if weight <= CASH_OUT_THRESHOLD:
                cash_weight += weight
                weights_df.loc[date, top_etfs[i]] = 0.0
            else:
                weights_df.loc[date, top_etfs[i]] = weight
        weights_df.loc[date, 'cash'] = cash_weight
        
        # Track runtime and show progress
        runtime = datetime.datetime.now() - start_time
        runtime_df.loc[date] = runtime.total_seconds()
        progress = (idx + 1) / len(data['prices'])
        print(f"\rProcessing {date}: {progress:.1%} [{runtime}]", end="")

    # Save results and plot runtime
    weights_df.to_csv(os.path.join(file_dir, "weight.csv"))
    runtime_df.plot()
    plt.show()

if __name__ == "__main__":
    calculate_portfolio_weights()