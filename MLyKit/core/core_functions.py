from pandas import DataFrame, read_csv, read_excel
from typing import Tuple


def load_data(filepath: str, filetype: str = "csv") -> DataFrame:
    """
    Loads a dataset from a file.

    Parameters:
        filepath (str): Path to the file.
        filetype (str): Type of file ('csv', 'excel'). Default is 'csv'.

    Returns:
        DataFrame: Loaded DataFrame.
    """
    if filetype == "csv":
        return read_csv(filepath)
    elif filetype == "excel":
        return read_excel(filepath)
    else:
        raise ValueError(
            "Unsupported filetype. Supported types are 'csv' and 'excel'."
        )


def save_data(df: DataFrame, filepath: str, filetype: str = "csv") -> None:
    """
    Saves a DataFrame to a file.

    Parameters:
        df (DataFrame): DataFrame to save.
        filepath (str): Path to save the file.
        filetype (str): Type of file ('csv', 'excel'). Default is 'csv'.

    Returns:
        None: The DataFrame is saved to disk.
    """
    if filetype == "csv":
        df.to_csv(filepath, index=False)
    elif filetype == "excel":
        df.to_excel(filepath, index=False)
    else:
        raise ValueError(
            "Unsupported filetype. Supported types are 'csv' and 'excel'."
        )


def train_test_split(
    df: DataFrame, ratio: float
) -> Tuple[DataFrame, DataFrame]:
    """
    Splits a DataFrame into training and testing sets.

    Parameters:
        df (DataFrame): DataFrame to split.
        ratio (float): Proportion of data to include in the training set.

    Returns:
        Tuple[DataFrame, DataFrame]: Training and testing DataFrames.
    """
    train_size = int(len(df) * ratio)
    train_set = df[:train_size]
    test_set = df[train_size:]
    return train_set, test_set
