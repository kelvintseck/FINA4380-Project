import pandas as pd
import numpy as np
import os
import scipy.optimize as sc
import sys
import math
import warnings
import datetime
from BoardMarketIndex import BoardMarketIndex

from numba import jit


"""
Input: list of etf name, 1d array of average returns
Output: sorted list of etf name with top and positveaverage return
"""
def sort_etf(etf_list, avg_return):
    idx = avg_return.argsort()[::-1]
    avg_return_sorted = avg_return.iloc[idx]
    filter_no = 0
    for index in range(len(etf_list)):
        filter_no = filter_no + 1
        if avg_return_sorted.iloc[index] == 0:
            break
    # sorted_list = etf_list[idx][:min(filter_no, 20)] 
    sorted_list = etf_list[idx][:filter_no] 
    return sorted_list

"""
Objective function to maximize Sharpe Ratio
"""

@jit(nopython=True)
def SR(w, mean, cov_mat):
    # w: weights (1D array), mean: mean returns (1D array), cov_mat: covariance matrix (2D array)
    vol = np.sqrt(w @ cov_mat @ w)  # Portfolio volatility
    sr = (mean @ w) / vol           # Sharpe Ratio
    return -sr                      # Negative for minimization

# Constraint function (not Numba-compiled)
def Constraint(w):
    return 1 - np.sum(w)  # Use np.sum for consistency

# Main Smart Beta function
def Function_SmartBeta(df):
    # Precompute mean and covariance as NumPy arrays
    mean = df.mean().to_numpy()      # Shape: (n_assets,)
    cov_mat = df.cov().to_numpy()    # Shape: (n_assets, n_assets)
    
    # Initial weights
    dn = len(df.columns)
    x_0 = np.ones(dn) / dn  # Equal weights
    
    # Bounds
    ubsmb = 0.4
    lbsmb = 0
    bndsa = [(lbsmb, ubsmb)] * dn  # List of tuples for each asset
    
    # Constraint
    cons = {'type': 'eq', 'fun': Constraint}
    
    # Tolerance and options
    tolerance = 1e-30
    options = {'maxiter': 100}
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Optimize
    res = sc.minimize(SR, x_0, args=(mean, cov_mat), method='SLSQP', 
                      bounds=bndsa, tol=tolerance, constraints=cons, 
                      options=options)
    
    return res.x  # Optimized weights

"""
@jit(nopython=True)
def SR(w, df):
    mean = np.array(df.mean())
    cov_mat = np.array(df.cov())
    w = np.array(w)
    vol = np.sqrt(w.T @ cov_mat @ w)     
    sr = mean @ w.T / vol
    return -sr

def Constraint(w):
    return (1 - sum(w))

def Function_SmartBeta(df):
    dn = len(df.columns)
    x_0 = np.ones(dn)/dn # inital weighting for SmartBeta Optimizer
    ubsmb = 0.4
    lbsmb = 0
    w = x_0
    bndsa = ((lbsmb, ubsmb),)
    tolerance = 1e-30
    for i in range(1, dn):
        bndsa = bndsa + ((lbsmb, ubsmb),)
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    res = sc.minimize(SR, x_0, args=(df,), method='SLSQP', bounds=bndsa, tol=tolerance, constraints=cons)#, options = {'maxiter': 100})
    B = res.x
    return B
"""

def get_weight():
    daily_price_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "ETFs_daily_prices.csv"), index_col=0)
    daily_return_df = daily_price_df.pct_change(fill_method=None).tail(-1)
    avg_return_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "average_returns.csv"), index_col=0)
    return_90days_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "returns_90d.csv"), index_col=0)
    return_90days_positive = pd.DataFrame(return_90days_df > 0)
    
    avg_return_90days_positive = return_90days_positive * avg_return_df
    etf_list = np.array(avg_return_df.columns)

    position_df = pd.DataFrame(columns=daily_price_df.columns, index=daily_price_df.index)
    weight_df = pd.DataFrame(float(0), columns=daily_price_df.columns, index=daily_price_df.index)
    #testcount = 0
    last_operation_time = datetime.datetime.now()
    for row_idx in range(len(daily_price_df)):
        avg_return = avg_return_df.iloc[row_idx]
        avg_return = avg_return.fillna(0)
        if sum(avg_return) == 0: #Skip date when average return is not yet ready
            continue
        #testcount = testcount + 1
        date = avg_return_df.index[row_idx]
        time1 = datetime.datetime.now()
        #print("Going to initiate class BoardMarketIndex")
        market_index = BoardMarketIndex(os.path.join(os.path.dirname(__file__), "ETFs_daily_prices.csv"), date)
        time2 = datetime.datetime.now()
        # print("Finish initiating. Time used is ", time2 - time1)
        strong_momentum = np.array([1 if (market_index.asset_is_in_strong_momentum_group(etf) == True) else 0 for etf in etf_list])
        #time3 = datetime.datetime.now()
        #print("Finish creating boolean array for BoardMarketIndex. Time used is ", time3 - time2)
        sorted_etf_list = sort_etf(etf_list, avg_return * strong_momentum)
        time4 = datetime.datetime.now()
        #print("Finish sorting ETFs. Time used is ", time4 - time3)
        daily_return_top_etf = daily_return_df[sorted_etf_list]
        weight = Function_SmartBeta(daily_return_top_etf)
        time5 = datetime.datetime.now()
        # print(f"Optimization done. Time used is ", time5 - time4)
        #if testcount == 200:
            #sys.exit()
        progress_bar = f"\r\rCurrent date: {date} Current process: {row_idx}/{len(daily_price_df)} [{'#' * int(row_idx * 100/ len(daily_price_df))}{' ' * (100 - int(row_idx * 100/ len(daily_price_df)))}] {int(row_idx * 100/ len(daily_price_df))}% Last operation took {datetime.datetime.now() - last_operation_time}"
        print(progress_bar, end="")
        for idx in range(len(weight)):
            weight_df.loc[date, sorted_etf_list[idx]] = weight[idx]
        last_operation_time = datetime.datetime.now()
    weight_df.to_csv(os.path.join(os.path.dirname(__file__), "weight.csv"))
        
if __name__ == "__main__":
    get_weight()
