import pandas as pd
import wholesale_customers as wc

def test_wholesale_customers():
    # Test read_csv_2 function
    data_file = 'data/wholesale_customers.csv'  # Adjust path as needed
    df = wc.read_csv_2(data_file)
    
    # Verify that Channel and Region columns are dropped
    print("Columns in the dataframe:", df.columns.tolist())
    
    # Verify that we have the 6 required numeric columns
    expected_columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    for col in expected_columns:
        if col not in df.columns:
            print(f"Warning: Expected column {col} not found")
    
    # Show the first few rows of the dataframe
    print("\nFirst 5 rows of the dataframe:")
    print(df.head())
    
    # Test summary_statistics function
    stats_df = wc.summary_statistics(df)
    
    # Verify that we have the required statistics
    print("\nSummary statistics:")
    print(stats_df)
    
    # Verify that mean and std are rounded to integers
    for col in ['mean', 'std']:
        if not all(isinstance(val, int) for val in stats_df[col]):
            print(f"Warning: {col} values are not all integers")

if __name__ == "__main__":
    test_wholesale_customers()