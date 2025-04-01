# %%
import yfinance as yf
import pandas as pd
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
    print(tickers)
    
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


