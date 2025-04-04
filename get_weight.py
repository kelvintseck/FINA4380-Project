import pandas as pd
import numpy as np
import os
import scipy.optimize as sc
"""
Input: list of etf name, 1d array of average returns
Output: sorted list of etf name with top average return
"""
def sort_etf(etf_list, avg_return):
    filter_no = 4 # Set number of etf wanted

    idx = avg_return.argsort()[::-1]
    sorted_list = etf_list[idx][:filter_no]
    return sorted_list

"""
Objective function to maximize Sharpe Ratio
"""
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
    lbsmb = 0.001
    w = x_0
    bndsa = ((lbsmb, ubsmb),)
    
    tolerance = 1e-30
    for i in range(1, dn):
        bndsa = bndsa + ((lbsmb, ubsmb),)
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    res = sc.minimize(SR, x_0, args=(df,), method='SLSQP', bounds=bndsa, tol=tolerance, constraints=cons)
    B = res.x
    return B

def get_weight():
    daily_price_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "ETFs_daily_prices.csv"), index_col=0)
    daily_return_df = daily_price_df.pct_change(fill_method=None).tail(-1)
    avg_return_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "average_returns.csv"), index_col=0)
    etf_list = np.array(avg_return_df.columns)

    position_df = pd.DataFrame(columns=daily_price_df.columns, index=daily_price_df.index)
    weight_df = pd.DataFrame(float(0), columns=daily_price_df.columns, index=daily_price_df.index)
    for row_idx in range(len(daily_price_df)):
        avg_return = avg_return_df.iloc[row_idx]
        avg_return = avg_return.fillna(0)
        if sum(avg_return) == 0: #Skip date when average return is not yet ready
            continue
        date = avg_return_df.index[row_idx]
        sorted_etf_list = sort_etf(etf_list, avg_return)
        daily_return_top_etf = daily_return_df[sorted_etf_list]
        weight = Function_SmartBeta(daily_return_top_etf)
        progress_bar = f"\r Current process: {int(round(row_idx * 100/ len(daily_price_df), 0))}% [{"#" * int(round(row_idx * 100/ len(daily_price_df), 0))}{" " * (100 - int(round(row_idx * 100/ len(daily_price_df), 0)))}]"
        print(progress_bar, end="")
        for idx in range(len(weight)):
            weight_df.loc[date, sorted_etf_list[idx]] = weight[idx]
    weight_df.to_csv(os.path.join(os.path.dirname(__file__), "weight.csv"))
        
if __name__ == "__main__":
    get_weight()