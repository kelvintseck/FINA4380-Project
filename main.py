if __name__ == '__main__':  
    pass
    """
    1. call functions in data_handler.py to download all the required .csv files
    2. For each symbol, create Stock object
    3. Let the date walks from the oldest one to someend  (this should be a loop)
        4. In data_handler, for each Stock, update the indicators
        5. In portfolio.py, create a new portfolio based on the updated information
        6. In strategy.py, compare the previous portfolio and the new portfolio, decide what to do (buy or hold or sell)
        7. In evaluation.py, record the performance
    8. In evaluation.py, visualize the performance
    """