import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


def plot_distribution(df: DataFrame, column: str) -> None:
    """
    Plots the distribution of a column.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column (str): The name of the column to plot.

    Returns:
        None
    """
    sns.histplot(df[column])
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_correlation_matrix(df: DataFrame) -> None:
    """
    Plots a correlation matrix.

    Parameters:
        df (DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def plot_time_series(df: DataFrame, time_col: str, value_col: str) -> None:
    """
    Plots a time series.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        time_col (str): The name of the time column.
        value_col (str): The name of the value column.

    Returns:
        None
    """
    plt.plot(df[time_col], df[value_col])
    plt.title(f"Time Series of {value_col} over {time_col}")
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.show()
