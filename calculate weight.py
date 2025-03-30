import pandas as pd
import numpy as np
import os
import scipy.optimize as sc

"""
Input: ETFs price time series dataframe, value of window
Output: Returns of ETFs on the latest date
"""
def cal_return(data, timeframe):
    row_today = len(data) - 1
    row_dayback = timeframe + 1
    return (data.iloc[row_today] - data.iloc[-row_dayback]) / data.iloc[-row_dayback]

"""
Input: ETFs price time series dataframe
Output: List of ETF tickers with top (specified number) momentum
"""
def get_sorted_list(df):
    return_90days = cal_return(df, 90)
    return_180days = cal_return(df, 180)
    return_360days = cal_return(df, 360)
    avg_return = (return_90days + return_180days + return_360days) / 3
    
    #Get etf with top average returns
    etf_number = 4
    etf_list = avg_return.index
    idx = np.argsort(avg_return)[::-1]
    sorted_etf_list = etf_list[idx][:etf_number]
    return sorted_etf_list

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
    ubsmb = 1 
    lbsmb = 0.01
    w = x_0
    bndsa = ((lbsmb, ubsmb),)
    
    tolerance = 1e-30
    for i in range(1, dn):
        bndsa = bndsa + ((lbsmb, ubsmb),)
    cons = (
        {'type': 'eq', 'fun': lambda w: 1 - sum(w)},
        {'type': 'ineq', 'fun': lambda w: np.sqrt(w.T @ w) - 0.5}
    )
    res = sc.minimize(SR, x_0, args=(df,), method='SLSQP', bounds=bndsa, tol=tolerance, constraints=cons)
    B = res.x
    return B
    
if __name__ == "__main__":
    daily_price = pd.read_csv(os.path.join(os.path.dirname(__file__), "ETFs_daily_prices.csv"), index_col=0)
    daily_price = daily_price.fillna(0)
    etf_effective_today = []
    for etf in daily_price.columns:
        if daily_price[etf].iloc[-1] != 0:
            etf_effective_today.append(etf)
    daily_price_effective_today = daily_price[etf_effective_today]
    daily_return = daily_price_effective_today.pct_change().tail(-1)
    
    etf_list = get_sorted_list(daily_price_effective_today)
    daily_return_top_etf = daily_return[etf_list]
    daily_return_top_etf.to_csv(os.path.join(os.path.dirname(__file__), "sorted_Etf.csv"))
    try:
        weight = Function_SmartBeta(daily_return_top_etf)
        print("Optimal weights:")
        for ticker, w in zip(etf_list, weight):
            print(f"{ticker}: {w:.2%}")
    except Exception as e:
        print(f"Error: {e}")