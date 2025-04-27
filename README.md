# FINA4380A Algorithmic Trading Strategies, Arbitrage and HFT - Trading Algorithm Project    
# Algorithmic Trading Strategy for ETF Portfolios Combining Momentum, Correlation, and Smart Beta Optimization 


## Process
1. **dataDownloader.py**: Filter and download price data of ETFs as *ETFs_daily_prices.csv*
2. **boardMarketIndex.py**: Read the historical data, separate the ETFs into different groups as an initial assets selection
3. **smartBeta.py**: Compute the new weights of each asset in the portfolio (and cash)
   - calls **boardMarketIndex.py** for each trading date. 
4. **portfolio.py**: Initialize a `portfolio` object by reading the *ETFs_daily_prices.csv* file. Simulate the trading by reading the assigned weights genereated by **smartBeta.py**.
5. **visualization.py**: Visualize the performance of a strategy, output .html file.
6. **comparison_visualization.py**: Visualize performances of different strategies.

Folder in the main branch contains the weights and results for variants.
- BMI (Market Breadth Index)
    - Variates of different thresholds of correlation and moving average window sizes.
- Momentum and Filter
    - Control cases where we remove Market Breadth Index filtering, momentum scoring, and 90-day return thresholds.
- Rebalance frequency
    - Variates of rebalance frequencies, including daily, ,weekly,  monthly, yearly. 
- Smart Beta
    - Sharpe ratio replaced by momentum returns, original Sharpe ratio, global minimum variance, risk parity, diversification ratio, 

- Comparison Plots 
    - Plots of performance metrics of intra-group strategies.
- runtime 
    - .csv files that record the running time of our **smartBeta.py** at each trading day (just for fun)


-----------------------------------------------------------------------------

### dataDownloader.py
- Filter and download price data of ETFs as *ETFs_daily_prices.csv*

### portfolio.py
- Contain class Portfolio for backtesting and recording the performance.
- Handle the changes in weights affected by the fluctuations of prices.

### boardMarketIndex.py
- Separate the ETFs into different groups as an initial assets selection
- Similar to MarketBreadth: prevent Momentum Crash by not investing in weak groups

### smartBeta.py
- Compute the new weights of each asset in the portfolio (and cash)    

### visualization.py
- Visualize the performance of the portfolio through bokeh, and output a html file.

### comparison_visualization.py
- Visualize performances of different strategies.
-----------------------------------------------------------------------------

### DCC-GARCH.ipynb
- Not in use due to fail in removing correlations between residuals
