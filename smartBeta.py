import pandas as pd
import numpy as np
import os
from scipy import optimize
import datetime
from boardMarketIndex import BoardMarketIndex
import matplotlib.pyplot as plt
from numba import jit
import warnings
import sys
# Constants
ROLLING_WINDOW = 360
MAX_WEIGHT = 0.4
MIN_WEIGHT = 0
CASH_OUT_THRESHOLD = 0.01
TOLERANCE = 1e-30
MAX_ITERATIONS = 100

#Switch for partial run
PARTIALPERIOD = 1825
PARTIAL = False #Base case: False

#Swithch for variants
VARIANT_SR_MU = False #Base Case = False; Variant: True
VARIANT_OPTIMAL_FUNCTION = 'RP' #Base case: whatever except variant input; Variants: 'RP', 'GMV'
VARIANT_TIME_INTERVAL = 0 #Base case: whatever except variant input; Variants: 'monthly', 'weekly'

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
    data['prices_monthly'] = data['prices'].resample('ME').last()
    data['prices_weekly'] = data['prices'].resample('W-MON').last()
    data['daily_returns'] = data['prices'].pct_change().tail(-1)
    data['monthly_returns'] = data['prices_monthly'].pct_change().tail(-1)
    data['weekly_returns'] = data['prices_weekly'].pct_change().tail(-1)
    return data

def get_top_etfs(etf_list, filtered_returns):
    """Sort ETFs by average returns and filter positive ones."""
    sorted_indices = filtered_returns.argsort()[::-1]
    sorted_returns = filtered_returns.iloc[sorted_indices]
    
    # Find first zero or negative return
    cutoff = (sorted_returns > 0).sum()
    return etf_list[sorted_indices][:cutoff]

@jit(nopython=True)
def calc_negative_sharpe(weights, means, cov_matrix, avg_returns, risk_free):
    """Sharpe Ratio optimal function for optimization."""
    if VARIANT_SR_MU == True:
        portfolio_return = np.dot(means, weights)
    else:
        portfolio_return = np.dot(avg_returns, weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(portfolio_return - risk_free) / portfolio_volatility

@jit(nopython=True)
def calc_glob_min_var(weights, cov_matrix):
    """Global Munimum Variance optimal function for optimization"""
    return weights.T @ cov_matrix @ weights

def calc_risk_parity(weights, cov_matrix):
    vol = np.sqrt(np.dot(np.dot(weights, cov_matrix), weights.T))
    marginal_contribution = np.dot(cov_matrix, weights.T) / vol
    r = (vol / weights.shape - weights * marginal_contribution.T)
    return np.dot(r, r.T)

def optimize_weights(rolling_returns, avg_returns, risk_free):
    """Optimize portfolio weights using Smart Beta strategy."""
    means = rolling_returns.mean().to_numpy()
    cov_matrix = rolling_returns.cov().to_numpy()
    num_assets = len(rolling_returns.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bndsa = [(MIN_WEIGHT, MAX_WEIGHT)] * num_assets
    cons = {'type': 'eq', 'fun': lambda w: 1 - np.sum(w)}
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    if VARIANT_OPTIMAL_FUNCTION == 'GMV':
        optimal_function = calc_glob_min_var
        args = (cov_matrix)
    elif VARIANT_OPTIMAL_FUNCTION == 'RP':
        optimal_function = calc_risk_parity
        args = (cov_matrix)
    else:
        optimal_function = calc_negative_sharpe
        args = (means, cov_matrix, avg_returns.to_numpy(), risk_free)

    result = optimize.minimize(
        optimal_function,
        initial_weights,
        args=args,
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
    initial_time = datetime.datetime.now()
    file_dir = os.path.dirname(__file__)
    data = load_data(file_dir)
    etf_list = np.array(data['avg_returns'].columns)
    weights_df = pd.DataFrame(0.0, columns=list(data['prices'].columns) + ['cash'], 
                            index=data['prices'].index)
    runtime_df = pd.DataFrame(columns=['Time'], index=data['prices'].index)

    if VARIANT_TIME_INTERVAL == 'monthly':
        df = data['prices_monthly']
        df_return = data['monthly_returns']
    elif VARIANT_TIME_INTERVAL == 'weekly':
        df = data['prices_weekly']
        df_return = data['weekly_returns']
    else:
        df = data['prices']
        df_return = data['daily_returns']

    if PARTIAL == True:
        df = df.iloc[:PARTIALPERIOD]
    
    for idx, date in enumerate(df.index):
        start_time = datetime.datetime.now()
        # Skip if no valid average returns
        valid = False
        while valid == False:
            try: #Adjust date when the date in weekly/ monthly returns does not fit with actual data file
                current_avg_returns = data['avg_returns'].loc[date].fillna(0)
                valid = True
            except:
                date -= datetime.timedelta(days=1)
        current_avg_returns = data['avg_returns'].loc[date].fillna(0) * data['returns_90d'].loc[date].fillna(0)
        if current_avg_returns.sum() <= 0: #Skip finding weights when no etf passes requirements
            continue
        try:
            risk_free_rate = data['risk_free'].loc[date].iloc[0]
        except:
            risk_free_rate = 0
        market_index = BoardMarketIndex(os.path.join(file_dir, "ETFs_daily_prices.csv"), os.path.join(file_dir, "average_momentum_returns.csv"), date.strftime('%Y-%m-%d'))
        
        # Identify ETFs with strong momentum
        momentum_flags = np.array([market_index.asset_is_in_strong_momentum_group(etf) 
                                 for etf in etf_list])
        filtered_return = current_avg_returns * momentum_flags
        if filtered_return.sum() <= 0:
            continue        
        # Get top performing ETFs
        top_etfs = get_top_etfs(etf_list, filtered_return)
        rolling_returns = get_rolling_data(df_return[top_etfs], idx)
        
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
        total_runtime = datetime.datetime.now() - initial_time
        runtime_df.loc[date] = runtime.total_seconds()
        progress = (idx + 1) / len(df)
        print(f"\rProcessing {date.strftime('%Y-%m-%d')}: {progress:.1%} [{'#' * int(progress * 100 + 1)}{' ' * (100 - int(progress * 100 + 1))}] Last runtime: {runtime} Total runtime: {total_runtime}", end="")

    # Save results and plot runtime
    weights_df.to_csv(os.path.join(file_dir, "weight.csv"))
    print("\nCreated weight.csv file")
    runtime_df.to_csv(os.path.join(file_dir, "runtime.csv"))

if __name__ == "__main__":
    calculate_portfolio_weights()