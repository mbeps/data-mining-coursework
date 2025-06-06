from typing import List

import pandas as pd
from pandas import Index
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore

# Part 1: Decision Trees with Categorical Attributes


# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file: str) -> DataFrame:
    """
    Read the adult dataset and drop the fnlwgt column.

    Args:
        data_file: Path to the adult dataset.
    Returns:
        The adult dataset.
    """
    df: DataFrame = pd.read_csv(data_file, na_values="?")
    if "fnlwgt" in df.columns:
        df = df.drop("fnlwgt", axis=1)
    return df


# Return the number of rows in the pandas dataframe df.
def num_rows(df: DataFrame) -> int:
    """
    Return the number of rows in the dataframe.

    Args:
        df: The pandas dataframe.
    Returns:
        The number of rows.
    """
    return len(df)


# Return a list with the column names in the pandas dataframe df.
def column_names(df: DataFrame) -> List[str]:
    """
    Return a list of column names in the dataframe.

    Args:
        df: The pandas dataframe.
    Returns:
        The column names.
    """
    return list(df.columns)


# Return the number of missing values in the pandas dataframe df.
def missing_values(df: DataFrame) -> int:
    """
    Return the total number of missing values in the dataframe.

    Args:
        df: The pandas dataframe.
    Returns:
        The total number of missing
    """
    return df.isna().sum().sum()


# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df: pd.DataFrame) -> List[str]:
    """
    Return list of column names that have at least one missing value.

    Args:
        df: The pandas dataframe.
    Returns:
        The column names with missing values.
    """
    return list(df.columns[df.isna().any()])


# Return the percentage of instances corresponding to persons whose education level is
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df: pd.DataFrame) -> float:
    """
    Calculate percentage of instances with education level Bachelors or Masters.
    Returns rounded to 1 decimal place.

    Args:
        df: The pandas dataframe.
    Returns:
        The percentage of instances with education level Bachelors or Masters.
    """
    education_count: int = df["education"].isin(["Bachelors", "Masters"]).sum()
    percentage: float = (education_count / len(df)) * 100
    return round(percentage, 1)


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df: DataFrame) -> DataFrame:
    """
    Remove all instances with at least one missing value from a copy of the dataframe.

    Args:
        df: The pandas dataframe.
    Returns:
        A copy of the dataframe without missing values.
    """
    df_copy: DataFrame = df.copy()
    return df_copy.dropna()


# Return a pandas dataframe (new copy) from the pandas dataframe df
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df: DataFrame) -> DataFrame:
    """
    Convert categorical attributes to numeric using one-hot encoding.
    Excludes the target attribute (class).

    Args:
        df: The pandas dataframe.
    Returns:
        The dataframe with categorical attributes converted to numeric using one-hot encoding.
    """
    # Drop the target column before encoding
    X: DataFrame = df.drop("class", axis=1)

    # Get numeric columns to exclude from encoding
    numeric_cols: Index[str] = X.select_dtypes(include=["int64", "float64"]).columns

    # One-hot encode all non-numeric columns
    encoded_df: DataFrame = pd.get_dummies(
        X, columns=[col for col in X.columns if col not in numeric_cols]
    )

    return encoded_df


# Return a pandas series (new copy), from the pandas dataframe df,
# containing only one column with the labels of the df instances
# converted to numeric using label encoding.
def label_encoding(df: DataFrame) -> Series:
    """
    Convert target labels to numeric using label encoding.

    Args:
        df: The pandas dataframe.
    Returns:
        The target labels converted to numeric using label encoding.
    """
    le = LabelEncoder()
    return pd.Series(le.fit_transform(df["class"]), index=df.index)


# Given a training set X_train containing the input attribute values
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train.
# Return a pandas series with the predicted values.
def dt_predict(X_train: DataFrame, y_train: Series) -> Series:
    """
    Build a decision tree and make predictions on training data.

    Args:
        X_train: The training set containing input attribute values.
        y_train: The training set containing labels for the training instances.
    Returns:
        The predicted labels.
    """
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return pd.Series(clf.predict(X_train), index=y_train.index)


# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.
def dt_error_rate(y_pred: pd.Series, y_true: pd.Series) -> float:
    """
    Calculate the error rate of predictions.

    Args:
        y_pred: The predicted labels.
        y_true: The true labels.
    Returns:
        The error rate.
    """
    return (y_pred != y_true).mean()
