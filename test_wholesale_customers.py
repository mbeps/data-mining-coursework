import pandas as pd
import numpy as np
import wholesale_customers as wc
import matplotlib.pyplot as plt

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
    
    # Use a smaller sample for faster testing
    df_sample = df.sample(n=min(50, len(df)), random_state=42)
    
    # Test standardize function
    print("\nTesting standardize function...")
    df_std = wc.standardize(df_sample)
    print("Mean of standardized data (should be close to 0):")
    print(df_std.mean())
    print("Std of standardized data (should be close to 1):")
    print(df_std.std())
    
    # Test kmeans function
    print("\nTesting kmeans function...")
    k = 3
    y_kmeans = wc.kmeans(df_sample, k)
    print(f"Kmeans cluster distribution for k={k}:")
    print(y_kmeans.value_counts())
    
    # Test kmeans_plus function
    print("\nTesting kmeans_plus function...")
    y_kmeans_plus = wc.kmeans_plus(df_sample, k)
    print(f"Kmeans++ cluster distribution for k={k}:")
    print(y_kmeans_plus.value_counts())
    
    # Test agglomerative function
    print("\nTesting agglomerative function...")
    y_agg = wc.agglomerative(df_sample, k)
    print(f"Agglomerative cluster distribution for k={k}:")
    print(y_agg.value_counts())
    
    # Test clustering_score function
    print("\nTesting clustering_score function...")
    score = wc.clustering_score(df_sample, y_kmeans)
    print(f"Silhouette score for kmeans: {score:.4f}")
    
    # Test cluster_evaluation function (with a very small sample to speed up testing)
    tiny_sample = df.sample(n=min(20, len(df)), random_state=42)
    print("\nTesting cluster_evaluation function (with a small sample)...")
    results = wc.cluster_evaluation(tiny_sample)
    print("Evaluation results shape:", results.shape)
    print("Evaluation results columns:", results.columns.tolist())
    print("Sample of evaluation results:")
    print(results.head())
    
    # Test best_clustering_score function
    print("\nTesting best_clustering_score function...")
    best_score = wc.best_clustering_score(results)
    print(f"Best silhouette score: {best_score:.4f}")
    
    # Test scatter_plots function (but don't actually display the plots)
    print("\nTesting scatter_plots function...")
    try:
        # Temporarily disable plot display
        plt.ioff()
        wc.scatter_plots(tiny_sample)
        plt.close('all')
        print("Scatter plots function ran without errors")
    except Exception as e:
        print(f"Error in scatter_plots function: {e}")
    finally:
        # Re-enable plot display
        plt.ion()

if __name__ == "__main__":
    test_wholesale_customers()