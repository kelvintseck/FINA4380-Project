import pandas as pd
import yfinance as yf

INDEX_FOR_MKTCON = "^GSPC"
INDEX_STARTDATE = "1929-01-01" # additional 1 year for moving average
INDEX_ENDDATE = "2025-03-01"

spx_data = yf.download(INDEX_FOR_MKTCON, start=INDEX_STARTDATE, end=INDEX_ENDDATE)
spx_close = spx_data[['Close']]
spx_close.to_csv("spx_close.csv")

spx_w_close = spx_data['Close'].resample('W').last()
spx_w_close.to_csv('spx_w_close.csv')

print(spx_close.columns) # not used yet, datatype issue
print(spx_w_close.columns)