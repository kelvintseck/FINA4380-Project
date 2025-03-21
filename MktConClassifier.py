import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any # enhances readability and helps with type checking, making it clearer what types of data are expected and returned.
import time

"""
input: csv file of market index
process: different method to classify market into (at least) 4 Market Conditions. methods including MA (direction and crossing) / MomCrashProb (??) / others...
output: dictionary: {"MC1": 1930-1933, 1940-1941; "MC2": 1933-1935; "MC3": ...; "MC4": ...}
output usage: for indentify factors strength in each MC
"""

"""
many strings seem can be set as variable, to prevent "hard codeing", but where should i set as something like global variables
"""

def calculate_time_used():
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

def Modified_df_MA_Vola(eq_df): # necessary
    # cal MA (d: (Long tf:150 Short:30); w = d/5: (Long:30 Short:6)
    eq_df["LongMA"] = eq_df["Close"].rolling(30).mean()
    eq_df["ShortMA"] = eq_df["Close"].rolling(6).mean()
    eq_df["Volatility"] = eq_df["Close"].pct_change(fill_method=None).rolling(6).std() # this Volatility is used only to confirm whether the change is valid or not
    new_eq_df = eq_df.dropna() # for rank
    return new_eq_df

## direction change (MA), cross (P, MA), position
def rankings(eq_df):
    """
    apply position rank for each time point; change of position indicate crossing -> potential action
    """
    eq_df['RankClose'] = None
    eq_df['RankShortMA'] = None
    eq_df['RankLongMA'] = None
    for t in eq_df.index:
        close_t, shortMA_t, longMA_t = eq_df.loc[t, 'Close'], eq_df.loc[t, 'ShortMA'], eq_df.loc[t, 'LongMA']
        rank = sorted([close_t, shortMA_t, longMA_t])
        eq_df.loc[t, 'RankClose'] = rank.index(close_t) + 1
        eq_df.loc[t, 'RankShortMA'] = rank.index(shortMA_t) + 1
        eq_df.loc[t, 'RankLongMA'] = rank.index(longMA_t) + 1

        # spx_w.loc[t, 'PosChange'] = []
        # if t !=  spx_w.index[0]:
        #     for ele in ["Close", "ShortMA", "LongMA"]:
        #         if spx_w.loc[t, f'Rank{ele}'] != spx_w.loc[t-1, f'Rank{ele}']:
        #             spx_w[t, 'PosChange'].append(ele)

def Calculate_trend_change(eq_df, name = "TrendChange"):
    """
    change of trend only LongMA is concerned (since "close" and ShortMA are consistently changing direction)
    same direction > 0; change < 0 -> True
    """
    eq_df[name] = ((eq_df["LongMA"] - eq_df["LongMA"].shift(1)) * (eq_df["LongMA"].shift(1) - eq_df["LongMA"].shift(2))) < 0 # test if trend continues
    # example: 100 99 then 98 -> (98-99)*(99-100)=1 same; 100 95 then 97 -> 2*-5=-10 change

def Calculate_crossing(eq_df, names: List[str]):
    """
    names=["Close_OverShortMA", "Close_OverLongMA", "ShortMA_OverLongMA"]
    1 cross above, -1 below, 0 no change
    """
    eq_df[names[0]] = None  # Above=1, Below=-1, None=0
    eq_df[names[1]] = None
    eq_df[names[2]] = None
    for i in range(len(eq_df.index)):
        if i != 0:
            t = eq_df.index[i]
            last = eq_df.index[i-1]
            close_t, shortma_t, longma_t = eq_df.loc[t, "Close"], eq_df.loc[t, "ShortMA"], eq_df.loc[t, "LongMA"]
            close_last, shortma_last, longma_last = eq_df.loc[last, "Close"], eq_df.loc[last, "ShortMA"], eq_df.loc[last, "LongMA"]
            # close over shortma
            if (close_t - shortma_t) * (close_last - shortma_last)<0: # rank changes
                if (close_t - shortma_t) > 0:
                    eq_df.loc[t, names[0]] = 1
                else:
                    eq_df.loc[t, names[0]] = -1
            else:
                eq_df.loc[t, names[0]] = 0
            # close over longma
            if (close_t - longma_t) * (close_last - longma_last)<0:
                if (close_t - longma_t) > 0:
                    eq_df.loc[t, names[1]] = 1
                else:
                    eq_df.loc[t, names[1]] = -1
            else:
                eq_df.loc[t, names[1]] = 0
            # shortma over longma
            if (shortma_t - longma_t) * (shortma_last - longma_last)<0:
                if (shortma_t - longma_t) > 0:
                    eq_df.loc[t, names[2]] = 1
                else:
                    eq_df.loc[t, names[2]] = -1
            else:
                eq_df.loc[t, names[2]] = 0


# def Classifier_cooldown(days: int):

def Classifier_MC_time(eq_df: pd.DataFrame, classifier: List[str]) -> [Dict[str, datetime], int]:
    """
    For 2 conditions we loop like this, but for several conditions (4 in our ideal case) we need to set rules
    """
    MC_time = None
    act_num_state = 0
    if classifier[0] == "TrendChange":
        if len(classifier) == 1:
            # initialize
            MC_time = {"MC1": [], "MC2": []}
            MC_time["MC1"].append(eq_df.index[0])
            last_t = eq_df.index[0]
            num_state = 1 # wo cd
            act_num_state = 1 # w cd

            classifier_cd = 0 # avoid constant change in volatile period

            for t in eq_df.index:
                if eq_df.loc[t, classifier[0]] == True: # change of state
                    num_state += 1 # change state regardless of cd
                    if (pd.to_datetime(t) - pd.to_datetime(last_t)).days > classifier_cd:
                        MC_time[f"MC{1 + num_state % 2}"].append(t)
                        last_t = t
                        act_num_state += 1

    elif classifier[0] == "ShortMA_OverLongMA":
        if len(classifier) == 1:
            # initialize
            MC_time = {"MC1": [], "MC2": []}
            MC_time["MC1"].append(eq_df.index[0])
            last_t = eq_df.index[0]
            num_state = 1 # wo cd
            act_num_state = 1 # w cd

            classifier_cd = 0 # avoid constant change in volatile period

            for t in eq_df.index:
                if eq_df.loc[t, classifier[0]] != 0: # change of state
                    num_state += 1 # change state regardless of cd
                    if (pd.to_datetime(t) - pd.to_datetime(last_t)).days > classifier_cd:
                        MC_time[f"MC{1 + num_state % 2}"].append(t)
                        last_t = t
                        act_num_state += 1

    return MC_time, act_num_state

def Plot_close_state(eq_df: pd.DataFrame, mc: Dict[str, datetime]):
    """
    plot closing price and indicate change of state with color.
    possible to add show_values = T/F for close, MAs
    """

    plt.figure(figsize=(14, 7))
    plt.plot(eq_df.index, eq_df['Close'], label='Closing Price', color='black')

    for state, dates in mc.items():
        for date in dates:
            plt.axvline(x=date, color='red' if state == 'MC1' else 'blue', linestyle='--') # , label=state
    plt.title('Closing Prices with Change of State Indicators')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(fname="Close_wShortLongMAcrossing")
    plt.show()

start_time = time.time()
def main():
    # target_index = spx_w
    # 1930-2014 as train, 2015-2024 as test; 10y first
    spx_w = pd.read_csv("spx_w_close.csv", index_col='Date')
    spx_w.rename(columns={spx_w.columns[0]: "Close"}, inplace=True)
    spx_w = spx_w.iloc[4001:4500] # testing for smaller period
    mc_classifier = ["ShortMA_OverLongMA"] # "TrendChange"

    # print(type(spx_w.index[0]))

    spx_w_mod = Modified_df_MA_Vola(spx_w)
    Calculate_trend_change(spx_w_mod, "TrendChange")
    Calculate_crossing(spx_w_mod, names=["Close_OverShortMA", "Close_OverLongMA", "ShortMA_OverLongMA"]) # order matters
    # print(spx_w_mod["TrendChange"].value_counts()) # count total states within the period we stated before
    print(spx_w_mod)
    mc_time, num_st = Classifier_MC_time(spx_w_mod, mc_classifier)

    Plot_close_state(spx_w_mod, mc_time)
    # rankings(spx_w)
    spx_w_mod.to_csv("spx_w_mod_ver3.csv")

    print(mc_time)
    print(num_st)
    calculate_time_used()

main()
