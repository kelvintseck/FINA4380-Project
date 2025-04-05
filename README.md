# FINA4380A Algorithmic Trading Strategies, Arbitrage and HFT - Trading Algorithm Project
Please directly upload the latest .py files or .csv files to the main branch.
Update this README.md accordingly if possible.

## Process
1. **dataDownloader.py**: Filter and download price data of ETFs as *ETFs_daily_prices.csv*
2. **portfolio.py**: Initialize a `portfolio` object by reading the *ETFs_daily_prices.csv* file
3. For each trading timepoint:    
    3.1. Get current or historical data (including prices, weights etc.) from `portfolio`    
    3.2. **boardMarketIndex.py**: Read the historical data, separate the ETFs into different groups as an initial assets selection   
    3.3. **smartBeta.py**: Compute the new weights of each asset in the portfolio (and cash)    
    3.4. **portfolio.py**: Simulate the trading with the given weights    
    3.5. `portfolio.advance_date()`
5. **evaluation.py**: Evaluate and visualize the performance




### main.py
- Centralized file for the whole process.

### dataDownloader.py
- Filter and download price data of ETFs as *ETFs_daily_prices.csv*

### portfolio.py
- Contain class Portfolio for backtesting and recording the performance.
- Handle the changes in weights affected by the fluctuations of prices.

### boardMarketIndex.py
- Separate the ETFs into different groups as an initial assets selection

### smartBeta.py
- Compute the new weights of each asset in the portfolio (and cash)    

### evaluation.py
- Display the final performance metrics.
- Visualize the performance of the portfolio through plotly.

### util.py
- Global variables for variates of our trading algorithm.
- Every .py files that may variate should import **util.py** and alter the computation accordingly.
