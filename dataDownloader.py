# %%
import yfinance as yf
import pandas as pd
import numpy as np
from yfinance import EquityQuery
from collections import OrderedDict

# %%
def rolling_calendar_returns(data, days):
    
    # Calculate returns
    returns = data.pct_change(periods=days)
    
    return returns

# %%
def strict_elementwise_average(*dfs):
    """
    Calculate element-wise average of multiple DataFrames, returning NaN if any value is missing
    
    Parameters:
    *dfs : variable number of pandas DataFrames with identical shapes
    
    Returns:
    pandas DataFrame with strict averaging (NaN if any input is NaN)
    """
    if not dfs:
        raise ValueError("At least one DataFrame must be provided")
    
    # Initialize with first DataFrame
    sum_df = dfs[0].copy()
    count_df = (~dfs[0].isna()).astype(int)
    
    # Accumulate sum and count
    for df in dfs[1:]:
        sum_df = sum_df.add(df, fill_value=0)
        count_df = count_df.add(~df.isna(), fill_value=0)
    
    # Calculate average only where count equals number of DataFrames
    total_dfs = len(dfs)
    avg_df = sum_df.where(count_df == total_dfs) / total_dfs
    return avg_df

# %%
def get_volume_matrix(tickers):
    """Return DataFrame with dates as index, tickers as columns, and volumes as values"""
    return yf.download(tickers, period="max", progress=False)["Volume"]

# %%
def get_price_matrix(tickers, period="max"):
    """Downloads historical prices, skips failed tickers, and returns DataFrame.
    
    Args:
        tickers: List of symbols
        period: Time period to download (default: max)
    
    Returns:
        DataFrame with dates as index, successful tickers as columns
    """
    successful_data = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False)["Close"]
            data.name = ticker  # Set column name to ticker
            successful_data.append(data)
        except Exception as e:
            print(f"✗ Failed {ticker}: {str(e).split('(')[0]}")  # Concise error
    
    return pd.concat(successful_data, axis=1)  # Combine all successful downloads

# Usage:
# ETFs_daily_prices = download_prices(all_etfs_list)

# %%
def extract_stock_list(sector_input):
    # Define the custom query
    custom_query = EquityQuery('and', [
        EquityQuery('eq', ['region', 'us']),
        EquityQuery('gte', ['dayvolume', 10000]),
        EquityQuery('gte', ['intradayprice', 5]),
         EquityQuery('eq', ['sector', sector_input]) # Exclude penny stocks
    ])

    response = yf.screen(custom_query, size = 250, sortField = 'intradaymarketcap', sortAsc = False)

    # Extract relevant fields from the response
    quotes = response['quotes']
    results = [
        {
            'ticker': stock['symbol'],
        }
        for stock in quotes
    ]

    # Extract tickers as a list
    tickers = [stock['symbol'] for stock in response['quotes']]
    print(f'Filtered stocks in {sector_input} are: {tickers}')
    
    return tickers

# Usage:
extract_stock_list('Healthcare')

# %%
def filter_liquid(tickers, min_turnover = 5e6):
    """Filters tickers by 20-day average dollar turnover (Volume * Close)."""
    data = yf.download(tickers, period="20d", progress=False)
    liquid = (data['Volume'] * data['Close']).mean() >= min_turnover
    return liquid[liquid].index.tolist()

# Usage:
liquid_stocks = filter_liquid(extract_stock_list('Healthcare'))

# %%
def get_correlation_matrix_and_returns(tickers, determine_correlation_period = "5Y", determine_return_period = "1Y"):
    """Fetch historical data and calculate correlation matrix and 1-year returns.
    
    Args:
        tickers (list): List of stock/ETF tickers to analyze
        
    Returns:
        tuple: (correlation_matrix, one_year_returns) 
               - correlation_matrix: DataFrame of pairwise correlations
               - one_year_returns: Series of 1-year returns for each ticker
    """
    
    # Download historical data to calcuate correlation
    price_for_correlation = get_price_matrix(tickers, period = determine_correlation_period)
    returns_correlation = price_for_correlation.pct_change().dropna(how = "all")
    corr_matrix = returns_correlation.corr()
    print(f'Price Matrix:\n {price_for_correlation}')
    print(f'Correlation Matrix:\n {corr_matrix}')
    
    # Download historical data to calcuate returns, drop ticker with lower returns if two tickers are highly correlated
    price_for_return = get_price_matrix(tickers, period = determine_return_period)
    n_year_returns = (price_for_return.iloc[-1] / price_for_return.iloc[0] - 1).dropna()

    # Calculate daily returns and correlation matrix


    # Calculate 1-year returns for each ticker
    # data_1y = returns_correlation.loc[start_date_1y:end_date]
    
    return corr_matrix, n_year_returns

def filter_high_correlation_tickers(tickers, max_correlation=0.3):
    """Filter out highly correlated tickers, keeping the best performer.
    
    Args:
        tickers (list): List of tickers to filter
        max_correlation (float): Threshold for considering correlation too high
        
    Returns:
        list: Filtered list of tickers with correlations below threshold
    """
    current_tickers = tickers.copy()
    removed_tickers = set()
    
    while True:
        # Get current tickers that are still present and not removed
        active_tickers = [t for t in current_tickers if t not in removed_tickers]
        if len(active_tickers) <= 1:
            break
        
        try:
            corr_matrix, one_year_returns = get_correlation_matrix_and_returns(active_tickers)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
        
        # Find pairs with correlation >= max_correlation
        high_corr_pairs = []
        n = len(active_tickers)
        for i in range(n):
            for j in range(i+1, n):
                ticker_i = active_tickers[i]
                ticker_j = active_tickers[j]
                corr = corr_matrix.loc[ticker_i, ticker_j]
                if abs(corr) >= max_correlation:
                    high_corr_pairs.append((ticker_i, ticker_j, corr))
        
        if not high_corr_pairs:
            break
        
        # Determine which tickers to remove
        to_remove = set()
        for pair in high_corr_pairs:
            ticker_i, ticker_j, _ = pair
            ret_i = one_year_returns.get(ticker_i, -np.inf)
            ret_j = one_year_returns.get(ticker_j, -np.inf)
            
            if ret_i < ret_j:
                to_remove.add(ticker_i)
            else:
                to_remove.add(ticker_j)
        
        if not to_remove:
            break
        
        removed_tickers.update(to_remove)
    
    filtered_tickers = [t for t in current_tickers if t not in removed_tickers]
    return filtered_tickers

# Example usage
# selected_etfs = ['SCHO', 'BIL', 'GBIL', 'SPTI', 'VTIP', 'SCHP', 'IGIB', 'HYG', 'BKLN', 'SRLN',
#                      'SNLN', 'FLBL', 'BWX', 'BNDX', 'EMLC', 'HYEM', 'EMHY', 'VTEB', 'TFI', 'MUNI',
#                      'HYMB', 'SHYD', 'HYD', 'IAU', 'USO', 'UNG', 'DBE', 'DBA', 'CORN', 'WEAT', 'SOYB',
#                      'CANE', 'DBB', 'JJM', 'COPX', 'REMX', 'UUP', 'FXY', 'FXB', 'CEW', 'BZF', 'FXCH',
#                      'EMFX', 'VIXY', 'SVOL', 'VXXB', 'ETHE', 'ARKB', 'RPAR', 'DIVB', 'AOR', 'VEGI',
#                      'XHB', 'ITB', 'VAW', 'DOW', 'CUT', 'FUND', 'JJU', 'FOIL', 'JJC', 'GDX', 'PPLT',
#                      'PALL', 'KOL', 'SLX', 'PSCL', 'CARZ', 'PBS', 'XRT', 'EMFM', 'PKG', 'PEJ', 'BITE',
#                      'RTH', 'LUXE', 'ONLN', 'FDIS', 'BETZ', 'JETS', 'TRYP', 'IAI', 'KRE', 'IXIS', 'FINX',
#                      'IAK', 'REZ', 'SRET', 'PBJ', 'FTXG', 'XLP', 'VICE', 'IBB', 'PJP', 'XPH', 'IHF',
#                      'IHI', 'XHE', 'QCLN', 'ICLN', 'XLU', 'XLE', 'AMLP', 'CRAK', 'URA', 'ITA', 'IYT',
#                      'SEA', 'IGV', 'SMH']
# filtered_tickers = filter_high_correlation_tickers(selected_etfs, 0.95)
# print(f'Number of tickers with low correlation: {len(filtered_tickers)}')
# print(f'Tickers with low correlation: {filtered_tickers}')

# %%
def rolling_momentum_returns(data, months_lookback=12, months_skip=1):
    """
    Calculate momentum returns by excluding the most recent month(s) and using a lookback window.
    
    Args:
        data: DataFrame with dates as index and prices as columns
        months_lookback: Total lookback period in months (e.g., 12 for annual momentum)
        months_skip: Number of recent months to exclude (e.g., 1 to exclude last month)
    
    Returns:
        DataFrame of momentum returns calculated as: 
        (price[t - skip] / price[t - skip - lookback] - 1)
    
    Example:
    --------
    # Calculate 12-month momentum excluding last month (2nd to 13th month)
    momentum_12m = rolling_momentum_returns(price_data, 12, 1)
    
    # Calculate 3-month momentum excluding last month (2nd to 4th month)
    momentum_3m = rolling_momentum_returns(price_data, 3, 1)    
    """
    # Convert months to trading days (approx 21 days/month)
    skip_days = 21 * months_skip
    lookback_days = 21 * months_lookback
    
    # Calculate returns using lookback window that excludes recent period
    shifted_start = data.shift(skip_days)
    shifted_end = data.shift(skip_days + lookback_days)
    
    returns = (shifted_start / shifted_end) - 1
    return returns

# %%
sector_etfs = {
    # Materials & Commodities
    "Agricultural_Inputs": ["VEGI", "MOO"],
    "Building_Materials": ["XHB", "ITB"],
    "Chemicals": ["VAW", "DOW"],
    "Specialty_Chemicals": ["FXZ"],
    "Lumber_Wood_Production": ["WOOD", "CUT"],
    "Paper_Products": ["FUND"],  # Part of broader materials ETFs
    "Aluminum": ["JJU", "FOIL"],
    "Copper": ["COPX", "JJC"],
    "Industrial_Metals_Mining": ["PICK", "REMX"],
    "Gold": ["GLD", "IAU", "GDX"], # Miners
    "Silver": ["SLV", "SIVR", "SIL"], # Miners
    "Precious_Metals_Mining": ["PPLT", "PALL", "SILJ"],
    "Coking_Coal": ["KOL"],  # Global coal ETF
    "Steel": ["SLX", "PSCL"],

    # Consumer Discretionary
    "Auto_Dealerships": ["CARZ"],
    "Auto_Manufacturers": ["CARZ", "VCR"],
    "Auto_Parts": ["CARZ"],
    "Recreational_Vehicles": ["PBS"],  # Subset of consumer cyclicals
    "Furnishings_Fixtures_Appliances": ["XHB", "PBS"],
    "Residential_Construction": ["ITB", "XHB"],
    "Textile_Apparel_Footwear": ["XRT", "EMFM"], # retail-heavy, emerging markets focus
    "Packaging_Containers": ["PKG", "XAR"],
    "Personal_Services": ["PEJ"],
    "Restaurants": ["BITE", "PEJ"],
    
    # Retail
    "Apparel_Retail": ["XRT", "RTH"],
    "Department_Stores": ["XRT", "RTH"],
    "Home_Improvement_Retail": ["XHB", "ITB"],
    "Luxury_Goods": ["LUXE", "ONLN"],
    "Internet_Retail": ["ONLN", "IBUY"],
    "Specialty_Retail": ["XRT", "FDIS"],
    
    # Leisure & Travel
    "Gambling": ["BETZ"],
    "Leisure": ["PEJ"],
    "Lodging": ["PEJ"],
    "Resorts_Casinos": ["BETZ"],
    "Travel_Services": ["JETS", "TRYP"],
    
    # Financials
    "Asset_Management": ["KCE", "IAI"],
    "Banks_Diversified": ["KBE"],
    "Banks_Regional": ["KRE", "QABA"],
    "Mortgage_Finance": ["REM", "MORT"],
    "Capital_Markets": ["KCE", "IAI"],
    "Financial_Data_Exchanges": ["IXIS", "FINX"],
    
    # Insurance
    "Insurance_Life": ["KIE", "IAK"],
    "Insurance_Property_Casualty": ["KIE", "IAK"],
    "Insurance_Reinsurance": ["KIE", "IAK"],
    "Insurance_Specialty": ["KIE", "IAK"],
    "Insurance_Brokers": ["KIE", "IAK"],
    "Insurance_Diversified": ["KIE", "IAK"],
    
    # Real Estate
    "Real_Estate_Development": ["REZ"],
    "Real_Estate_Services": ["REZ"],
    "REIT_Healthcare": ["REZ"],
    "REIT_Hotel_Motel": ["REZ"],
    "REIT_Industrial": ["REZ"],
    "REIT_Office": ["REZ"],
    "REIT_Residential": ["REZ"],
    "REIT_Retail": ["REZ"],
    "REIT_Mortgage": ["REM"],
    "REIT_Specialty": ["SRET"],
    "REIT_Diversified": ["VNQ", "XLRE"],
    
    # Consumer Staples
    "Beverages_Brewers": ["BIB"],
    "Beverages_Wineries_Distilleries": ["BIB"],
    "Beverages_Non_Alcoholic": ["PBJ"],
    "Confectioners": ["PBJ", "FTXG"],
    "Farm_Products": ["DBA", "VEGI"],
    "Household_Personal_Products": ["XLP"],
    "Tobacco": ["XLP", "VICE"],
    
    # Healthcare
    "Biotechnology": ["IBB", "XBI"],
    "Drug_Manufacturers_General": ["PJP", "XPH"],
    "Drug_Manufacturers_Specialty": ["PJP", "XPH"],
    "Healthcare_Plans": ["IHF"],
    "Medical_Care_Facilities": ["IHF"],
    "Medical_Devices": ["IHI", "XHE"],
    "Pharmaceutical_Retailers": ["XPH"],
    
    # Utilities
    "Utilities_Renewable": ["QCLN", "ICLN"],
    "Utilities_Regulated": ["XLU", "VPU"],
    
    # Energy
    "Oil_Gas_Drilling": ["XOP"],
    "Oil_Gas_EP": ["XOP"],
    "Oil_Gas_Integrated": ["XLE"],
    "Oil_Gas_Midstream": ["AMLP"],
    "Oil_Gas_Refining": ["CRAK"],
    "Thermal_Coal": ["KOL"],
    "Uranium": ["URA", "URNM"],
    
    # Industrials
    "Aerospace_Defense": ["ITA", "PPA"],
    "Industrial_Machinery": ["XAR"],
    "Railroads": ["XTN", "IYT"],
    "Shipping": ["SEA"],
    
    # Technology
    "Software": ["IGV", "WCLD"],
    "Semiconductors": ["SOXX", "SMH"],
    "Solar": ["TAN", "ICLN"]
}

# Example: Access all Biotech ETFs
biotech_etfs = sector_etfs["Biotechnology"]
print(f"Biotech ETFs: {biotech_etfs}")

# Example: Download all Gold-related ETFs
gold_etfs = sector_etfs["Gold"]
print(f"Gold ETFs: {gold_etfs}")

# %%
ficc_etfs = {
    # ======== U.S. TREASURIES ========
    "Treasuries": {
        "Short_Term": ["SHY", "VGSH", "SCHO", "BIL", "GBIL"],
        "Intermediate": ["IEI", "IEF", "VGIT", "GOVT", "SPTI"],
        "Long_Term": ["TLT", "VGLT", "SPTL", "EDV", "TMF"],
        "TIPS": ["TIP", "VTIP", "SCHP", "LTPZ", "STIP"]
    },
    
    # ======== CORPORATE BONDS ========
    "Corporate_Bonds": {
        "Investment_Grade": ["LQD", "VCIT", "VCSH", "IGIB", "IGLB"],
        "High_Yield": ["HYG", "JNK", "HYLB", "ANGL", "HYBB"],
        "Bank_Loans": ["BKLN", "SRLN", "SNLN", "FLBL"]
    },
    
    # ======== INTERNATIONAL BONDS ========
    "International_Bonds": {
        "Developed_Markets": ["BWX", "IGOV", "BNDX", "IGOV"],
        "Emerging_Markets": ["EMB", "PCY", "EMLC", "HYEM", "EMHY"]
    },
    
    # ======== MUNICIPAL BONDS ========
    "Municipal_Bonds": {
        "National": ["MUB", "VTEB", "TFI", "MUNI"],
        "High_Yield_Munis": ["HYMB", "SHYD", "HYD"]
    },
    
    # ======== COMMODITIES ======== 
    "Commodities": {
        "Precious_Metals": ["GLD", "IAU", "SLV", "GLTR", "SGOL"],
        "Energy": ["USO", "BNO", "UNG", "UCO", "DBE"],
        "Agriculture": ["DBA", "CORN", "WEAT", "SOYB", "CANE"],
        "Industrial_Metals": ["DBB", "JJM", "COPX", "REMX"]
    },
    
    # ======== CURRENCIES ========
    "Currencies": {
        "Major_FX": ["UUP", "UDN", "FXE", "FXY", "FXB"],
        "Emerging_FX": ["CEW", "BZF", "FXCH", "EMFX"]
    },
    
    # ======== SPECIALTY FICC ========
    "Specialty": {
        "Volatility": ["VIXY", "SVOL", "VXXB"],  # VXX replaced with VXXB
        "Crypto": ["BITO", "ETHE", "BTF", "ARKB"],
        "Multi_Asset": ["RPAR", "GTIP", "DIVB", "AOR"]
    }
}

# Example: Access all Treasury ETFs
treasury_etfs = [etf for sublist in ficc_etfs["Treasuries"].values() for etf in sublist]
print(f"All Treasury ETFs: {treasury_etfs}")

# Example: Download all commodity ETFs
commodity_etfs = [etf for sublist in ficc_etfs["Commodities"].values() for etf in sublist]
print(f"Commodity ETFs: {commodity_etfs}")

# %%
# Simple flattening with list comprehension
sector_etfs_list = [
    etf 
    for subcategory in sector_etfs.values() 
    for etf in subcategory
]

# Remove duplicates while preserving order
sector_etfs_list = list(OrderedDict.fromkeys(sector_etfs_list))

print(f"Total Sectors ETFs: {len(sector_etfs_list)}")  # Count of unique ETFs

# %%
ficc_etfs_list = [
    etf 
    for asset_class in ficc_etfs.values() 
    for subcategory in asset_class.values() 
    for etf in subcategory
]

# Remove duplicates while preserving order
ficc_etfs_list = list(OrderedDict.fromkeys(ficc_etfs_list))

print(f"Total FICC ETFs: {len(ficc_etfs_list)}")

# %%
# Combines ficc_etfs_list and sector_etfs_list into a single list with:
# 1. All duplicates removed (keeps only first occurrence)
# 2. Original order preserved (first appearance order)
# Example: If 'GLD' exists in both lists, only the first occurrence is kept
all_etfs_list = list(OrderedDict.fromkeys(ficc_etfs_list + sector_etfs_list))
print(f"Total ETFs: {len(all_etfs_list)}")

# %%
ETFs_daily_prices = get_price_matrix(all_etfs_list, '20Y')

# %%
returns_1d = rolling_calendar_returns(ETFs_daily_prices, 1)
returns_90d = rolling_calendar_returns(ETFs_daily_prices, 90)
returns_180d = rolling_calendar_returns(ETFs_daily_prices, 180)
returns_360d = rolling_calendar_returns(ETFs_daily_prices, 360)

returns_90d.to_csv('returns_90d.csv')
returns_180d.to_csv('returns_180d.csv')
returns_360d.to_csv('returns_360d.csv')
ETFs_daily_prices.to_csv('ETFs_daily_prices.csv')
strict_elementwise_average(returns_90d, returns_180d, returns_360d).to_csv('average_returns.csv')

# %%
#Testing below, please ignore or delete=================================================================================

# %%
filtered_tickers = filter_high_correlation_tickers(all_etfs_list, 0.95)
print(f'Number of tickers with low correlation: {len(filtered_tickers)}')
print(f'Tickers with low correlation: {filtered_tickers}')

# %%
sector_mapping ={
    "Commincation": "Communication Services",
    "Technology": 'Technology',
    'Real Estate': 'Global Real Estate'
}

# %%
sector_test = (yf.Ticker("SOXX").info)['category']
display((yf.Ticker("RWO").info)['category'])
display((yf.Ticker("TSM").info)['sector'])

display((yf.Ticker("SOXX").info))



# sector_list = []

# for i in ['SOXX','REET','XLF']:
#        extract_stock_list((yf.Ticker(i).info)['category']) 

# sector_test = ['Technology','Healthcare']

# # for i in sector_test:
# #     extract_stock_list(i)


