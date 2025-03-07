import pandas as pd
from typing import List
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore

# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    """
    Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
    data_file will be populated with the string 'wholesale_customers.csv'.
    """
    # Read the CSV file
    df = pd.read_csv(data_file)
    
    # Drop the Channel and Region columns
    df = df.drop(['Channel', 'Region'], axis=1)
    
    return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    """
    Return a pandas dataframe with summary statistics of the data.
    Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
    These strings index the new dataframe columns.
    Each row should correspond to an attribute in the original data and be indexed with the attribute name.
    """
    # Calculate statistics
    mean_values = df.mean().round().astype(int)  # Round to closest integer
    std_values = df.std().round().astype(int)    # Round to closest integer
    min_values = df.min()
    max_values = df.max()
    
    # Create a new dataframe with the summary statistics
    summary_df = pd.DataFrame({
        'mean': mean_values,
        'std': std_values,
        'min': min_values,
        'max': max_values
    })
    
    return summary_df

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    """
    Given a dataframe df with numeric values, return a dataframe (new copy)
    where each attribute value is subtracted by the mean and then divided by the
    standard deviation for that attribute.
    """
    df_standardized = df.copy()
    
    # Standardize each column
    for column in df_standardized.columns:
        mean = df_standardized[column].mean()
        std = df_standardized[column].std()
        df_standardized[column] = (df_standardized[column] - mean) / std
    
    return df_standardized


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans(df, k):
    """
    Given a dataframe df and a number of clusters k, return a pandas series y
    specifying an assignment of instances to clusters, using kmeans.
    y should contain values in the set {0,1,...,k-1}.
    To see the impact of the random initialization,
    using only one set of initial centroids in the kmeans run.
    """
    from sklearn.cluster import KMeans # type: ignore
    
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(df)
    y = pd.Series(kmeans_model.labels_, index=df.index)
    
    return y


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    """
    Given a dataframe df and a number of clusters k, return a pandas series y
    specifying an assignment of instances to clusters, using kmeans++.
    y should contain values from the set {0,1,...,k-1}.
    """
    from sklearn.cluster import KMeans
    
    # Initialize KMeans with k clusters and kmeans++ initialization
    # In scikit-learn, init='k-means++' is the default
    kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans_model.fit(df)
    y = pd.Series(kmeans_model.labels_, index=df.index)
    
    return y

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def agglomerative(df, k):
    """
    Given a dataframe df and a number of clusters k, return a pandas series y
    specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
    y should contain values from the set {0,1,...,k-1}.
    """
    from sklearn.cluster import AgglomerativeClustering
    
    agg = AgglomerativeClustering(n_clusters=k)
    agg.fit(df)
    y = pd.Series(agg.labels_, index=df.index)
    return y

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def clustering_score(X, y):
    """
    Given a data set X and an assignment to clusters y
    return the Silhouette score of this set of clusters.
    """
    from sklearn.metrics import silhouette_score
    
    # The silhouette score is only defined if k is between 2 and n-1 where n is the number of samples
    if len(set(y)) <= 1 or len(set(y)) >= len(X):
        return 0.0
    
    return silhouette_score(X, y)


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def cluster_evaluation(df):
    """
    Perform the cluster evaluation described in the coursework description.
    Given the dataframe df with the data to be clustered,
    return a pandas dataframe with an entry for each clustering algorithm execution.
    Each entry should contain the: 
    'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
    'data' type: either 'Original' or 'Standardized',
    'k': the number of clusters produced,
    'Silhouette Score': for evaluating the resulting set of clusters.
    """
    from sklearn.cluster import KMeans

    df_standardized = standardize(df)    
    k_values = [3, 5, 10]
    results = []
    
    for k in k_values:
        # For each data type (original and standardized)
        for data_type, data in [('Original', df), ('Standardized', df_standardized)]:
            # Run agglomerative clustering once
            y_agg = agglomerative(data, k)
            score_agg = clustering_score(data, y_agg)
            results.append({
                'Algorithm': 'Agglomerative',
                'data': data_type,
                'k': k,
                'Silhouette Score': score_agg
            })
            
            # Run k-means 10 times with different initializations
            for i in range(10):
                # Create a new KMeans instance with a different random_state for each run
                kmeans_model = KMeans(n_clusters=k, random_state=i)
                kmeans_model.fit(data)
                y_kmeans = pd.Series(kmeans_model.labels_, index=data.index)
                score_kmeans = clustering_score(data, y_kmeans)
                results.append({
                    'Algorithm': 'Kmeans',
                    'data': data_type,
                    'k': k,
                    'Silhouette Score': score_kmeans
                })
    
    # Convert the results to a pandas DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def best_clustering_score(rdf):
    """
    Given the performance evaluation dataframe produced by the cluster_evaluation function,
    return the best computed Silhouette score.
    """
    # Find the maximum Silhouette score
    best_score = rdf['Silhouette Score'].max()
    
    return best_score


# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    """
    Run the Kmeans algorithm with k=3 by using the standardized data set.
    Generate a scatter plot for each pair of attributes.
    Data points in different clusters should appear with different colors.
    Each plot is saved to a file in the project root directory.
    """
    import matplotlib.pyplot as plt
    import itertools
    import os
    
    df_standardized = standardize(df)    
    y = kmeans(df_standardized, 3)
    
    attributes = df.columns
    attribute_pairs = list(itertools.combinations(attributes, 2))
    
    for attr1, attr2 in attribute_pairs:
        plt.figure(figsize=(8, 6))
        
        # Plot points with different colors based on their cluster
        for cluster_id in sorted(y.unique()):
            cluster_points = df_standardized[y == cluster_id]
            plt.scatter(
                cluster_points[attr1], 
                cluster_points[attr2], 
                label=f'Cluster {cluster_id}',
                alpha=0.7
            )
        
        plt.xlabel(attr1)
        plt.ylabel(attr2)
        plt.title(f'{attr1} vs {attr2}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        filename = f"cluster_plot_{attr1}_vs_{attr2}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory