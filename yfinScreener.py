import yfinance as yf
from yfinance import EquityQuery

def extract_stock_list():
    # Define the custom query
    custom_query = EquityQuery('and', [
        EquityQuery('gte', ['intradaymarketcap', 1000000000]),  # Market cap >= $1B
        EquityQuery('eq', ['region', 'us']),
        EquityQuery('gte', ['dayvolume', 50000]),               # Volume >= 50K shares
        EquityQuery('gt', ['intradayprice', 10]),               # Price > $10
        # EquityQuery('gt', ['dilutedeps1yrgrowth.lasttwelvemonths', 0])       # EPS > 0
    ])

    # Run the screen, sorting by market cap descending
    response = yf.screen(custom_query, size=100, sortField='intradaymarketcap', sortAsc=False)
    print(response)

    # Extract relevant fields from the response
    quotes = response['quotes']
    results = [
        {
            'ticker': stock['symbol'],
            'price': stock['regularMarketPrice'],
            'volume': stock['regularMarketVolume'],
        }
        for stock in quotes
    ]

    # Print the results
    for result in results:
        print(f"Ticker: {result['ticker']}, Price: ${result['price']}")

    # Extract tickers as a list
    tickers = [stock['symbol'] for stock in response['quotes']]

    # Print the list for verification
    print("Ticker list for yf.download:", tickers)
