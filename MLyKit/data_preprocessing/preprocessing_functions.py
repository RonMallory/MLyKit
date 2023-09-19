import pandas as pd
from pandas import DataFrame
from typing import List, Tuple, Dict, Union, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler


def fill_string_col(
    df: DataFrame, cols: List[str], default: str = "Unknown"
) -> None:
    """
    Fills missing values in specified string columns with a default value.

    Parameters:
        df (DataFrame): The DataFrame to modify.
        cols (List[str]): List of column names to fill.
        default (str): The default value to fill missing values with.

    Returns:
        None: The DataFrame is modified in place.
    """
    for col in cols:
        df[col].fillna(default, inplace=True)


def fill_most_common(df: DataFrame, cols: List[str]) -> None:
    """
    Fills missing values in specified columns with the most common value in each column. # noqa E501

    Parameters:
        df (DataFrame): The DataFrame to modify.
        cols (List[str]): List of column names to fill.

    Returns:
        None: The DataFrame is modified in place.
    """
    for col in cols:
        most_common = df[col].mode()[0]
        df[col].fillna(most_common, inplace=True)


def fill_average_round(df: DataFrame, cols: List[str]) -> None:
    """
    Fills missing values in specified columns with the rounded average value of each column. # noqa E501

    Parameters:
        df (DataFrame): The DataFrame to modify.
        cols (List[str]): List of column names to fill.

    Returns:
        None: The DataFrame is modified in place.
    """
    for col in cols:
        average_rounded = round(df[col].mean())
        df[col].fillna(average_rounded, inplace=True)


def downcast_dataframe(df: DataFrame) -> None:
    """
    Downcast the columns of a Pandas DataFrame to the most efficient data types. # noqa E501

    Parameters:
        df (DataFrame): The DataFrame to modify.

    Returns:
        None: The DataFrame is modified in place.
    """
    # Downcast int and float types
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        else:
            # Handle other data types if needed
            pass

    # Downcast object types to category if unique values are less than 50% of total values # noqa E501
    for col in df.select_dtypes(include=["object"]).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype("category")


def label_encode(
    df: DataFrame, cols: List[str], inplace: bool = False
) -> Union[None, Tuple[DataFrame, Dict[str, LabelEncoder]]]:
    """
    Applies label encoding to the specified columns of a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to modify.
        cols (List[str]): List of column names to encode.
        inplace (bool): Whether to modify the DataFrame in place. Default is False. # noqa E501

    Returns:
        Union[None, Tuple[DataFrame, Dict[str, LabelEncoder]]]: If inplace is True, returns None. Otherwise, returns a tuple containing # noqa E501
        the modified DataFrame and a dictionary mapping column names to their respective LabelEncoder instances. # noqa E501
    """
    label_encoders = {}
    if inplace:
        target_df = df
    else:
        target_df = df.copy()

    for col in cols:
        le = LabelEncoder()
        target_df[col] = le.fit_transform(target_df[col].astype(str))
        label_encoders[col] = le

    if inplace:
        return None
    else:
        return target_df, label_encoders


def separate_data(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Separates a DataFrame into two DataFrames based on the presence of missing values. # noqa E501

    Parameters:
        df (DataFrame): The original DataFrame to separate.

    Returns:
        Tuple[DataFrame, DataFrame]: A tuple containing two DataFrames.
        The first DataFrame includes rows with no missing values.
        The second DataFrame includes rows with at least one missing value.
    """
    df_no_missing = df.dropna()
    df_with_missing = df[df.isna().any(axis=1)]
    return df_no_missing, df_with_missing


def scale_features(
    df: DataFrame, target_col: Optional[str] = None
) -> Tuple[DataFrame, StandardScaler]:
    """
    Scales the features of a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to scale.
        target_col (Optional[str]): The target column to exclude from scaling. Default is None. # noqa E501

    Returns:
        Tuple[DataFrame, StandardScaler]: A tuple containing the scaled DataFrame and the scaler used for scaling. # noqa E501
    """
    scaler = StandardScaler()

    # Separate target if it exists
    if target_col and target_col in df.columns:
        target = df[target_col]
        features = df.drop(columns=[target_col])
    else:
        target = None
        features = df

    # Scale features
    features_scaled = scaler.fit_transform(features)

    # Create a new DataFrame with scaled features (and target, if applicable)
    df_scaled = DataFrame(features_scaled, columns=features.columns)
    if target is not None:
        df_scaled[target_col] = target

    return df_scaled, scaler
