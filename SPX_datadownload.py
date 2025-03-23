import pandas as pd
import yfinance as yf

# input
INDEX_FOR_MKTCON = "^GSPC" # also possible for list of stocks
INDEX_STARTDATE = "1950-01-01"
INDEX_ENDDATE = "2025-03-01"
TRAIN_INDEX_STARTDATE = "1950-01-01" # volume appears only after 1950
TRAIN_INDEX_ENDDATE = "2014-12-31" # ?can we do cross-validation?
target_index_columns = ['Close', 'Volume']
# output file name
INDEX_CLOSE_FILENAME = 'spx_d_close.csv'
INDEX_VOLUME_FILENAME = "spx_d_vol.csv"

"""
1. Set input for concerned 1) index or equities; 2) period of date; 3) value wanting to extract
2. Set output file name
3. Run program to extract index data, and output as csv
p.s. Only for Day interval
p.s. seperate spx training data and total (test) data if possible

# use the following code to read output file for cleaniness:
import SPX_datadownload
spx_close = pd.read_csv(SPX_datadownload.INDEX_CLOSE_FILENAME, index_col='Date', header=0)
spx_vol = pd.read_csv(SPX_datadownload.INDEX_CLOSE_FILENAME, index_col='Date', header=0)
"""

# Start here.
spx_data = yf.download(INDEX_FOR_MKTCON, start=TRAIN_INDEX_STARTDATE, end=TRAIN_INDEX_ENDDATE)[target_index_columns]

# close
spx_d_close = spx_data[target_index_columns[0]].resample('D').last().dropna()
spx_d_close.rename(columns={spx_d_close.columns[0]: target_index_columns[0]}, inplace=True)
spx_d_close.to_csv(INDEX_CLOSE_FILENAME)

# volume
spx_d_vol = spx_data[target_index_columns[1]].resample('D').last().dropna()
spx_d_vol.rename(columns={spx_d_vol.columns[0]: target_index_columns[1]}, inplace=True)
spx_d_vol.to_csv(INDEX_VOLUME_FILENAME)
