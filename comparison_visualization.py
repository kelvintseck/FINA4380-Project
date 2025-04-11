import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import defaultdict

def plot_dataframe_comparison(df_list, attribute, save_path, plot_type='line', title=None, xlabel=None, ylabel=None, labels=None):
    """
    Plot a comparison graph for a list of DataFrames based on a specified attribute.
    
    Parameters:
    - df_list (list): List of pandas DataFrames, each containing the attribute to plot.
    - attribute (str): The column name in DataFrames to compare.
    - save_path (str): the path of the folder that the plots should be saved.
    - plot_type (str): Type of plot ('line', 'bar', 'scatter'). Default is 'line'.
    - title (str, optional): Title of the plot.
    - xlabel (str, optional): Label for x-axis.
    - ylabel (str, optional): Label for y-axis.
    - labels (list, optional): List of labels for each DataFrame in the legend.
    
    
    Returns:
    - None: Save the plot.
    """
    # Input validation
    if not df_list or not all(isinstance(df, pd.DataFrame) for df in df_list):
        raise ValueError("df_list must be a non-empty list of pandas DataFrames")
    if not attribute:
        raise ValueError("attribute must be a non-empty string")
    
    # Check if attribute exists in all DataFrames
    for i, df in enumerate(df_list):
        if attribute not in df.columns:
            raise ValueError(f"Attribute '{attribute}' not found in DataFrame at index {i}")
    
    # Set default labels if none provided
    if labels is None:
        labels = [f'DataFrame {i+1}' for i in range(len(df_list))]
    elif len(labels) != len(df_list):
        raise ValueError("Length of labels must match length of df_list")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot each DataFrame
    for df, label in zip(df_list, labels):
        if plot_type == 'line':
            plt.plot(df.index, df[attribute], label=label)
        elif plot_type == 'bar':
            plt.bar(df.index, df[attribute], label=label, alpha=0.5)
        elif plot_type == 'scatter':
            plt.scatter(df.index, df[attribute], label=label)
        else:
            raise ValueError("plot_type must be 'line', 'bar', or 'scatter'")
    
    # Customize plot
    plt.title(title if title else f'Comparison of {attribute}')
    plt.xlabel(xlabel if xlabel else 'Index')
    plt.ylabel(ylabel if ylabel else attribute)
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)



if __name__ == "__main__":
    folder_path = os.path.dirname(__file__)
    groups = defaultdict(list)
    groupsList = {"BMI", "Momentum and Filter", "Rebalance frequency", "Smart Beta"}
    attributes = ["Cumulative Return", "Annualized Return", "Annualized Volatility", "VaR", "Expected Shortfall", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "value"]

    saved_folder_path = os.path.join(folder_path, "Comparison Plots")
    
    for group in groupsList:
        path_to_folder = os.path.join(folder_path, group)
        csvFileList = os.listdir(path_to_folder)
        for filename in csvFileList:
            if filename.endswith('.csv') and filename.startswith('performance'):
                file = os.path.join(path_to_folder, filename)
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                groups[group].append((df, filename.lstrip("performance_metrics_").rstrip(".csv")))
                
        for attribute in attributes:
            plot_dataframe_comparison(
                df_list=[item[0] for item in groups[group]],
                attribute=attribute,
                plot_type='line',
                title=f'Comparison of {attribute} Across DataFrames',
                xlabel='Date',
                ylabel=f'{attribute}',
                labels=[item[1] for item in groups[group]],
                save_path=saved_folder_path
            )