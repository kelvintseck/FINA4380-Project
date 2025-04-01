import pandas
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

"""
aim to get optimal exit dd% for each etf
or spectrum for gradual exit
"""

# 1. Get ETF data # seem no need training or test, just research anyways
def get_etf_data(tickers, start_date="2020-01-01", end_date=datetime.today().strftime('%Y-%m-%d')):
    data = yf.download(tickers, period="max", interval="1d", auto_adjust=False)["Adj Close"]
    return data

# Replace cummax due to recency effect, ppl tend to forgot tops that were years ago
def recent_cummax(series, lookback_period='365D'):
    recent_data = series[series.index >= (series.index.max() - pd.Timedelta(lookback_period))]
    return recent_data.cummax()

# Modified function to calculate and plot maximum drawdown
def calculate_max_drawdown(price_series):

    roll_max = price_series.cummax() # returns the maximum value encountered up to that point.
    # Calculate drawdown
    drawdown = price_series / roll_max - 1.0
    # Calculate maximum drawdown
    max_drawdown = drawdown.min()

    return abs(max_drawdown), drawdown

def all_asset_return_after_drawdown_table(price_series, tickers=[]):
    for ticker in tickers:
        per_asset_return_after_drawdown_table(price_series[ticker], ticker = ticker)

def per_asset_return_after_drawdown_table(price_series, max_dd_tolerance = 0.1, ticker = 0):
    dd_series = calculate_max_drawdown(price_series)[1]
    asset_ts = pd.concat([price_series, dd_series], axis=1)
    asset_ts.columns = ['Price', 'Drawdown']
    print(asset_ts)

    # initialize
    tiles = [75,50,25] # ,10,5
    req_insight = ["count", "avg_p_1w", "avg_p_2w", "avg_p_1m", "avg_p_2m", "tile75_1w", "tile50_1w", "tile25_1w", "tile75_2w", "tile50_2w", "tile25_2w", "tile75_1m", "tile50_1m", "tile25_1m", "tile75_2m", "tile50_2m", "tile25_2m"] # for %tile, only care about 1m at the moment
    dd_ret_table_df = pandas.DataFrame(index=[i / 100 for i in range(1, int(max_dd_tolerance * 100) + 1)], columns=req_insight)
    print(dd_ret_table_df)

    dd_ret_record = {}
    time_frames = {
        '1w': pd.Timedelta(weeks=1),
        '2w': pd.Timedelta(weeks=2),
        '1m': pd.Timedelta(weeks=4),  # Approximation of 1 month
        '2m': pd.Timedelta(weeks=8)  # Approximation of 2 months
    }
    for time_frame, timedelta in time_frames.items():
        dd_ret_record[time_frame] = [[] for _ in range(int(max_dd_tolerance * 100))]
    print(dd_ret_record)

    # MAIN
    drawdown_dates = [[] for i in range(int(max_dd_tolerance*100))]
    recording = True

    # Recording dd dates
    for i in range(int(max_dd_tolerance*100)): # loop through k% drawdown (1-10)
        dd_focus = -(i+1)/100
        for date, drawdown in asset_ts["Drawdown"].items():
            if drawdown <= dd_focus and recording:  # Drawdown hits -1%
                drawdown_dates[i].append(date)
                recording = False
            elif drawdown == 0 and not recording:  # Turn the camera back on (Drawdown returns to 0%)
                recording = True
        print(drawdown_dates[i])
        dd_ret_table_df.loc[-dd_focus, "count"] = len(drawdown_dates[i])

    print(drawdown_dates)
    print(dd_ret_table_df["count"])

    # use dd dates and project perf afterwards
    for i in range(int(max_dd_tolerance*100)): # range(1): #
        for dddate in drawdown_dates[i]: # loop thru the dates in current i
            for time_frame, timedelta in time_frames.items(): # fill 0.1 dd for all tf first
                future_date = dddate + timedelta
                # skip if future data is nan
                if future_date not in asset_ts.index:
                    continue
                # while future_date not in asset_ts.index: # avoid geting Nan price
                #     future_date += pd.Timedelta(days=1)
                future_price = asset_ts.loc[future_date, 'Price']
                current_price = asset_ts.loc[dddate, 'Price']
                proj_ret = (future_price - current_price) / current_price
                dd_ret_record[time_frame][i].append(float(proj_ret)) # record 1w return after dd dates
    # drawdown_dates no more use.
    print(dd_ret_record)

    # take average for every tf and monthly return tile
    for i in range(int(max_dd_tolerance*100)): # range(1,2):
        dd = (i+1)/100
        for time_frame, ret_list in dd_ret_record.items():
            dd_ret_table_df.loc[dd, f"avg_p_{time_frame}"] = sum(ret_list[i]) / len(ret_list[i]) # remind that ret list is list of list

            # # %tile (monthly only)
            # if time_frame == '1w':
            for tile in tiles:
                tile_ret = np.percentile(ret_list[i], tile)
                dd_ret_table_df.loc[dd, f"tile{tile}_{time_frame}"] = tile_ret

    print(dd_ret_table_df)

    # export csv to see if any problem
    dd_ret_table_df.to_csv(f"{ticker}_dd_insight_all.csv", index=True)
    pass

def determine_optimal_k_or_spectrum():
    pass

if __name__ == "__main__":
    etf_tickers = ['XLF', 'GlD']

    tickers = etf_tickers
    price_data = get_etf_data(tickers)

    all_asset_return_after_drawdown_table(price_data, tickers=price_data.columns) # finish the class first
    # per_asset_return_after_drawdown_table(price_data["GLD"])