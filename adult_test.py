import pandas as pd
import numpy as np
import os
from typing import List
from pandas.core.frame import DataFrame
from pandas.core.series import Series

# Import all functions from the fixed implementation
from adult import (
    read_csv_1, 
    num_rows, 
    column_names, 
    missing_values, 
    columns_with_missing_values, 
    bachelors_masters_percentage,
    data_frame_without_missing_values,
    one_hot_encoding,
    label_encoding,
    dt_predict,
    dt_error_rate
)

def print_separator(title: str) -> None:
    """Print a separator with a title for better readability."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "-"))
    print("="*80)

def run_tests() -> None:
    """Run tests on all the implemented functions"""
    filepath = './data/adult.csv'
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        print("Please ensure the adult.csv file is in the data folder.")
        return
    
    print_separator("TASK 1: LOADING AND EXPLORING DATA")
    
    # Test read_csv_1
    print("Loading the dataset...")
    df = read_csv_1(filepath)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    
    # Verify fnlwgt column is dropped
    if 'fnlwgt' in df.columns:
        print("WARNING: fnlwgt column was not dropped!")
    else:
        print("fnlwgt column was correctly dropped")
    
    # Test num_rows
    count = num_rows(df)
    print(f"\nNumber of instances: {count}")
    
    # Test column_names
    cols = column_names(df)
    print(f"\nColumns in the dataset ({len(cols)}):")
    for i, col in enumerate(cols, 1):
        print(f"  {i}. {col}")
    
    # Test missing_values
    missing = missing_values(df)
    print(f"\nNumber of missing values: {missing}")
    
    # Test columns_with_missing_values
    cols_missing = columns_with_missing_values(df)
    print(f"\nColumns with missing values ({len(cols_missing)}):")
    if cols_missing:
        for col in cols_missing:
            count = df[col].isna().sum()
            print(f"  - {col}: {count} missing values")
    else:
        print("  No columns with missing values found")
    
    # Test bachelors_masters_percentage
    percentage = bachelors_masters_percentage(df)
    print(f"\nPercentage of Bachelors and Masters: {percentage}%")
    
    print_separator("TASK 2: DATA PREPROCESSING")
    
    # Test data_frame_without_missing_values
    df_clean = data_frame_without_missing_values(df)
    original_count = num_rows(df)
    clean_count = num_rows(df_clean)
    removed = original_count - clean_count
    print(f"Rows before removing missing values: {original_count}")
    print(f"Rows after removing missing values: {clean_count}")
    print(f"Removed {removed} rows ({(removed/original_count)*100:.2f}% of data)")
    
    # Test one_hot_encoding
    X_encoded = one_hot_encoding(df_clean)
    print(f"\nOne-hot encoded features shape: {X_encoded.shape}")
    print(f"Number of features after encoding: {X_encoded.shape[1]}")
    print("\nSample of encoded features (first 5 columns):")
    first_cols = min(5, X_encoded.shape[1])
    print(X_encoded.head(2).iloc[:, :first_cols])
    
    # Test label_encoding
    y_encoded = label_encoding(df_clean)
    unique_values = y_encoded.unique()
    print(f"\nEncoded labels shape: {y_encoded.shape}")
    print(f"Unique values in encoded labels: {unique_values}")
    print(f"Label distribution: {y_encoded.value_counts().to_dict()}")
    
    print_separator("TASK 3: DECISION TREE CLASSIFICATION")
    
    # Train decision tree and get predictions
    predictions = dt_predict(X_encoded, y_encoded)
    
    # Calculate error rate
    error = dt_error_rate(predictions, y_encoded)
    
    print(f"Decision tree training error rate: {error:.4f}")
    print(f"Decision tree training accuracy: {1-error:.4f}")
    
    # Compare with expected distribution
    print(f"\nPrediction distribution: {predictions.value_counts().to_dict()}")
    print(f"Actual distribution: {y_encoded.value_counts().to_dict()}")

if __name__ == "__main__":
    run_tests()