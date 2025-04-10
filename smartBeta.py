import pandas as pd
import numpy as np
import os
from scipy import optimize
import datetime
from boardMarketIndex import BoardMarketIndex
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

# Switch for partial run
PARTIALPERIOD = 30
PARTIAL = False  # Base case: False

# Swithch for variants
VARIANT_SR_MU = False  # Base Case = False; Variant: True
VARIANT_OPTIMAL_FUNCTION = 'SR'  # Base case: 'SR'; Variants: 'RP', 'GMV', 'DR'
VARIANT_TIME_INTERVAL = 'quarterly'  # Base case: 'daily'; Variants: 'monthly', 'weekly', 'quarterly', 'yearly'
VARIANT_REMOVE_MOMENTUM = False # Base case: False; Variant: True # To ignore momentum return in calculation
VARIANT_REMOVE_90DAYS = False # Base case: False; Variant: True
VARIANT_REMOVE_BMI = True # Base case: False; Variant: True # To ignore BroadMarketIndex in calculation
BMI_LEADER_SEARCH_BY = "highest_abs_correlation" # Base Case = "highest_abs_correlation"; Var: 'highest_momentum_score'
BMI_CORRELATION_THRESHOLD_FOR_GRP = 0.7 # Base Case = 0.5; Var = 0.5, 0.6
BMI_MA_PERIOD = 60 # Base Case = 60; Var = 200

def variant_case():
    case_name = ""
    if VARIANT_SR_MU == True:
        case_name += "_SR_MU"
    if VARIANT_OPTIMAL_FUNCTION != 'SR':
        case_name += f"_{VARIANT_OPTIMAL_FUNCTION}"
    if VARIANT_TIME_INTERVAL != 'daily':
        case_name += f"_{VARIANT_TIME_INTERVAL}"
    if VARIANT_REMOVE_MOMENTUM == True:
        case_name += "_XMomum"
    if VARIANT_REMOVE_BMI == True:
        case_name += "_XBMI"
    if BMI_LEADER_SEARCH_BY != 'highest_abs_correlation':
        case_name += "_MomumScore"
    if BMI_CORRELATION_THRESHOLD_FOR_GRP != 0.7:
        case_name += f"_GrpCorr{BMI_CORRELATION_THRESHOLD_FOR_GRP}"
    if BMI_MA_PERIOD != 60:
        case_name += "_200MA"

    match case_name:
        case "":
            case_name = "_Base"
        case "_SR_MU_Xmomentum_XBMI":
            case_name = "_control_case"
    return case_name

def load_data(file_dir):
    """Load all required data files."""
    files = {
        'prices': 'ETFs_daily_prices.csv',
        'avg_returns': 'average_momentum_returns.csv',
        'returns_90d': 'returns_90d.csv',
        'risk_free': 'riskFree.csv'
    }
    warnings.filterwarnings('ignore', category=UserWarning)
    data = {key: pd.read_csv(os.path.join(file_dir, filename), index_col=0, parse_dates=True, dayfirst=True, date_format='%Y-%m-%d')
            for key, filename in files.items()}
    # Preprocess data
    data['returns_90d'] = data['returns_90d'].fillna(0) >= 0
    data['prices_yearly'] = data['prices'].resample('YE').last()    
    data['prices_quarterly'] = data['prices'].resample('QE').last()    
    data['prices_monthly'] = data['prices'].resample('ME').last()
    data['prices_weekly'] = data['prices'].resample('W-MON').last()
    data['yearly_returns'] = data['prices_yearly'].pct_change().tail(-1)
    data['quarterly_returns'] = data['prices_quarterly'].pct_change().tail(-1)
    data['monthly_returns'] = data['prices_monthly'].pct_change().tail(-1)
    data['weekly_returns'] = data['prices_weekly'].pct_change().tail(-1)
    data['daily_returns'] = data['prices'].pct_change().tail(-1)
    return data

def get_top_etfs(etf_list, filtered_returns):
    """Sort ETFs by average returns and filter positive ones."""
    sorted_indices = filtered_returns.fillna(0).argsort()[::-1]
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

@jit(nopython=True)
def calc_risk_parity(weights, cov_matrix):
    vol = np.sqrt(np.dot(np.dot(weights, cov_matrix), weights.T))
    marginal_contribution = np.dot(cov_matrix, weights.T) / vol
    r = (vol / weights.shape - weights * marginal_contribution.T)
    return np.dot(r, r.T)

def calc_negative_diversification_ratio(weights, cov_matrix):
    numerator = np.sqrt(np.diag(cov_matrix)) @ weights.T
    denominator = np.sqrt(weights.T @ cov_matrix @ weights)
    return - numerator / denominator

def optimize_weights(rolling_returns, current_returns, risk_free):
    """Optimize portfolio weights using Smart Beta strategy."""
    num_assets = len(current_returns)
    means = rolling_returns.fillna(0).mean().to_numpy()
    cov_matrix = rolling_returns.fillna(0).cov().to_numpy()
    initial_weights = np.ones(num_assets) / num_assets
    bndsa = [(MIN_WEIGHT, MAX_WEIGHT)] * num_assets
    cons = {'type': 'eq', 'fun': lambda w: 1 - np.sum(w)}
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    match VARIANT_OPTIMAL_FUNCTION:
        case 'GMV':
            optimal_function = calc_glob_min_var
            args = (cov_matrix)
        case 'RP':
            optimal_function = calc_risk_parity
            args = (cov_matrix)
        case 'DR':
            optimal_function = calc_negative_diversification_ratio
            args = (cov_matrix) 
        case 'SR':
            optimal_function = calc_negative_sharpe
            args = (means, cov_matrix, current_returns.to_numpy(), risk_free)

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

@jit(nopython=True)
def adjust_weight(df, date_list):
    row_count = -1
    for row in df.index:
        row_count += 1
        if row not in date_list:
            new_row = df.index[row_count - 1]
            df.loc[row] = df.loc[new_row]
    return df

def calculate_portfolio_weights():
    """Main function to calculate and save portfolio weights."""
    initial_time = datetime.datetime.now()
    file_dir = os.path.dirname(__file__)
    data = load_data(file_dir)
    etf_list = np.array(data['avg_returns'].columns)
    weights_df = pd.DataFrame(0.0, columns=list(data['prices'].columns) + ['cash'],
                              index=data['prices'].index)
    weights_df = weights_df.assign(cash = float(1))
    runtime_df = pd.DataFrame(columns=['Time'], index=data['prices'].index)
    print("Current mode is " + variant_case())

    match VARIANT_TIME_INTERVAL:
        case 'yearly':
            df = data['prices_yearly']
            df_return = data['yearly_returns']
        case 'quarterly':
            df = data['prices_quarterly']
            df_return = data['quarterly_returns']    
        case 'monthly':
            df = data['prices_monthly']
            df_return = data['monthly_returns']
        case 'weekly':
            df = data['prices_weekly']
            df_return = data['weekly_returns']
        case 'daily':
            df = data['prices']
            df_return = data['daily_returns']

    if PARTIAL == True:
        df = df.iloc[:PARTIALPERIOD]
        df_return = df_return.iloc[:PARTIALPERIOD]
        weights_df = weights_df.iloc[:PARTIALPERIOD]
        runtime_df = runtime_df.iloc[:PARTIALPERIOD]

    # For non-daily time interval
    date_list = []

    for idx, date in enumerate(df.index):
        start_time = datetime.datetime.now()

        # Adjust date when the date in weekly/ monthly returns does not fit with actual data file
        valid = False
        while valid == False:
            try: 
                lookup = data['prices'].loc[date]
                date_list.append(date)
                valid = True
            except:
                date -= datetime.timedelta(days=1)

        rolling_returns = get_rolling_data(df_return, idx)
        try: # Skip first day as there is no return item on first day
            NONNA_Count = rolling_returns.loc[date].count().sum()
        except:
            continue
        if NONNA_Count < 2: # Should have at least 2 ETFs to construct covariance matrix  
            continue      
        # For control case
        means = rolling_returns.mean()

        if VARIANT_REMOVE_MOMENTUM == False: # Filter for ETF with non-positive 90days return
            if VARIANT_REMOVE_90DAYS == False:
                current_returns = data['avg_returns'].loc[date] * data['returns_90d'].loc[date]
            else: 
                current_returns = data['avg_returns'].loc[date]
        else: # Use Rolling return instead of momentum return for remove momentum Variant case
            current_returns = means
        if current_returns.fillna(0).sum() <= 0:  # Skip finding weights when no ETF passes requirements
            continue

        try: # Find the effective risk free rate for the time interval of investing
            match VARIANT_TIME_INTERVAL:
                case 'yearly':
                    time = 1
                case 'quarterly':
                    time = 4
                case 'monthly':
                    time = 12
                case 'weekly':
                    time = 52
                case 'daily':
                    time = 365

            risk_free_rate = data['risk_free'].loc[date].iloc[0] / 100 / time
        except:
            risk_free_rate = 0
       
        if VARIANT_REMOVE_BMI == False: # Filter for ETF that does not belongs to strong momentum group according to BMI
            market_index = BoardMarketIndex(os.path.join(file_dir, "ETFs_daily_prices.csv"),
                                            os.path.join(file_dir, "average_momentum_returns.csv"),
                                            date.strftime('%Y-%m-%d'),
                                            leader_search_by=BMI_LEADER_SEARCH_BY,
                                            correlation_threshold_for_grp=BMI_CORRELATION_THRESHOLD_FOR_GRP,
                                            ma_period=BMI_MA_PERIOD)

            # Identify ETFs with strong momentum
            momentum_flags = np.array([market_index.asset_is_in_strong_momentum_group(etf)
                                       for etf in etf_list])
            filtered_return = current_returns * momentum_flags
        else:
            filtered_return = current_returns

        if filtered_return.fillna(0).sum() <= 0: # Skip finding weights if performance of stock is bad
            continue

        # Get top performing ETFs
        top_etfs = get_top_etfs(etf_list, filtered_return)
        if len(top_etfs) <= 1: # Do not invest if there is no suitable ETF; At least invest in two ETFs
            continue

        # Optimize weights
        optimal_weights = optimize_weights(rolling_returns[top_etfs], current_returns[top_etfs], risk_free_rate)

        # Handle small weights and cash allocation
        weight_sum = 0
        for i, weight in enumerate(optimal_weights):
            if weight <= CASH_OUT_THRESHOLD:
                weights_df.loc[date, top_etfs[i]] = 0
            else:
                weights_df.loc[date, top_etfs[i]] = weight
                weight_sum += weight
        weights_df.loc[date, 'cash'] = 1 - weight_sum
        # Track runtime and show progress
        runtime = datetime.datetime.now() - start_time
        total_runtime = datetime.datetime.now() - initial_time
        runtime_df.loc[date] = runtime.total_seconds()
        progress = (idx + 1) / len(df)
        print( f"\rProcessing {date}: {progress:.1%} [{'#' * int(progress * 100)}{' ' * (100 - int(progress * 100))}] Last runtime: {runtime} Total runtime: {total_runtime}",
              end="")

    if VARIANT_TIME_INTERVAL != 'daily':
        print("\nAdjusting weights for non-trading date", end="")
        weights_df = adjust_weight(weights_df, date_list)

    # Create output files for weights and runtime
    weight_filename = "weight" + variant_case() + ".csv"
    weights_df.to_csv(os.path.join(file_dir, weight_filename))
    runtime_filename = "runtime" + variant_case() + ".csv"
    runtime_df.to_csv(os.path.join(file_dir, runtime_filename))
    print("\nCreated weight and runtime csv files")

if __name__ == "__main__":
    calculate_portfolio_weights()