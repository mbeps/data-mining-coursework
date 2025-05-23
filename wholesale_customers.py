import itertools
from typing import List
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pandas import Index
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.cluster import AgglomerativeClustering, KMeans  # type: ignore
from sklearn.metrics import silhouette_score  # type: ignore

# Part 2: Cluster Analysis


# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file: str) -> DataFrame:
    """
    Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
    data_file will be populated with the string 'wholesale_customers.csv'.

    Args:
        data_file: The path to the data file.
    Returns:
        A pandas dataframe containing the data set.
    """
    df: DataFrame = pd.read_csv(data_file)
    df = df.drop(["Channel", "Region"], axis=1)
    return df


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns.
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df: DataFrame) -> DataFrame:
    """
    Return a pandas dataframe with summary statistics of the data.
    Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
    These strings index the new dataframe columns.
    Each row should correspond to an attribute in the original data and be indexed with the attribute name.

    Args:
        df: The pandas dataframe containing the data set.
    Returns:
        A pandas dataframe with summary statistics of the data.
    """
    # Calculate statistics
    mean_values: Series = df.mean().round().astype(int)
    std_values: Series = df.std().round().astype(int)
    min_values: Series = df.min()
    max_values: Series = df.max()

    # Create a new dataframe with the summary statistics
    summary_df = pd.DataFrame(
        {"mean": mean_values, "std": std_values, "min": min_values, "max": max_values}
    )

    return summary_df


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df: DataFrame) -> DataFrame:
    """
    Given a dataframe df with numeric values, return a dataframe (new copy)
    where each attribute value is subtracted by the mean and then divided by the
    standard deviation for that attribute.

    Args:
        df: The pandas dataframe containing the data set.
    Returns:
        A new pandas dataframe with standardized values.
    """
    df_standardized: DataFrame = df.copy()
    return (df_standardized - df_standardized.mean()) / df_standardized.std()


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans(df: DataFrame, k: int) -> Series:
    """
    Given a dataframe df and a number of clusters k, return a pandas series y
    specifying an assignment of instances to clusters, using kmeans.
    y should contain values in the set {0,1,...,k-1}.
    Uses only one set of initial centroids to observe initialization impact.

    Args:
        df: The pandas dataframe containing the data set.
        k: The number of clusters.
    Returns:
        A pandas series specifying an assignment of instances to clusters.
    """
    kmeans_model = KMeans(n_clusters=k, init="random", n_init=1, random_state=42)
    kmeans_model.fit(df)
    return pd.Series(kmeans_model.labels_, index=df.index)


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df: DataFrame, k: int) -> Series:
    """
    Given a dataframe df and a number of clusters k, return a pandas series y
    specifying an assignment of instances to clusters, using kmeans++.
    y should contain values from the set {0,1,...,k-1}.

    Args:
        df: The pandas dataframe containing the data set.
        k: The number of clusters.
    Returns:
        A pandas series specifying an assignment of instances to clusters.
    """
    # Initialize KMeans with k clusters and kmeans++ initialization
    # In scikit-learn, init='k-means++' is the default
    kmeans_model = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans_model.fit(df)
    return pd.Series(kmeans_model.labels_, index=df.index)


# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def agglomerative(df: DataFrame, k: int) -> Series:
    """
    Given a dataframe df and a number of clusters k, return a pandas series y
    specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
    y should contain values from the set {0,1,...,k-1}.

    Args:
        df: The pandas dataframe containing the data set.
        k: The number of clusters.
    Returns:
        A pandas series specifying an assignment of instances to clusters.
    """
    agg = AgglomerativeClustering(n_clusters=k)
    agg.fit(df)
    return pd.Series(agg.labels_, index=df.index)


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the:
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative',
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def clustering_score(X: DataFrame, y: Series) -> float:
    """
    Given a data set X and an assignment to clusters y
    return the Silhouette score of this set of clusters.

    Args:
        X: The pandas dataframe containing the data set.
        y: A pandas series specifying an assignment of instances to clusters.
    Returns:
        The Silhouette score of the set of clusters.
    """
    # The silhouette score is only defined if k is between 2 and n-1 where n is the number of samples
    if len(set(y)) <= 1 or len(set(y)) >= len(X):
        return 0.0

    return silhouette_score(X, y)


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def cluster_evaluation(df: DataFrame) -> DataFrame:
    """
    Perform the cluster evaluation described in the coursework description.
    Given the dataframe df with the data to be clustered,
    return a pandas dataframe with an entry for each clustering algorithm execution.
    Each entry should contain the:
    'Algorithm' name: either 'Kmeans' or 'Agglomerative',
    'data' type: either 'Original' or 'Standardized',
    'k': the number of clusters produced,
    'Silhouette Score': for evaluating the resulting set of clusters.

    Args:
        df: The pandas dataframe containing the data set.
    Returns:
        A pandas dataframe with the clustering evaluation results.
    """
    df_standardized: DataFrame = standardize(df)
    k_values: List[int] = [3, 5, 10]
    results: list[dict[str, Any]] = []

    for k in k_values:
        # For each data type (original and standardized)
        for data_type, data in [("Original", df), ("Standardized", df_standardized)]:
            # Run agglomerative clustering once
            y_agg: Series = agglomerative(data, k)
            score_agg: float = clustering_score(data, y_agg)
            results.append(
                {
                    "Algorithm": "Agglomerative",
                    "data": data_type,
                    "k": k,
                    "Silhouette Score": score_agg,
                }
            )

            # Run k-means 10 times with different initializations
            for i in range(10):
                y_kmeans: Series = kmeans(data, k)
                score_kmeans: float = clustering_score(data, y_kmeans)
                results.append(
                    {
                        "Algorithm": "Kmeans",
                        "data": data_type,
                        "k": k,
                        "Silhouette Score": score_kmeans,
                    }
                )

    # Convert the results to a pandas DataFrame
    return pd.DataFrame(results)


def best_clustering_score(rdf: DataFrame) -> float:
    """
    Given the performance evaluation dataframe produced by the cluster_evaluation function,
    return the best computed Silhouette score.

    Args:
        rdf: The pandas dataframe with the clustering evaluation results.
    Returns:
        The best computed Silhouette score.
    """
    return rdf["Silhouette Score"].max()


# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df: DataFrame) -> None:
    """
    Run the Kmeans algorithm with k=3 by using the standardized data set.
    Generate a scatter plot for each pair of attributes.
    Data points in different clusters should appear with different colors.
    Each plot is saved to a file in the project root directory.

    Args:
        df: The pandas dataframe containing the data set.
    Returns:
        None
    """
    df_standardized: DataFrame = standardize(df)
    y: Series = kmeans(df_standardized, 3)

    attributes: Index[str] = df.columns
    attribute_pairs = list(itertools.combinations(attributes, 2))

    for attr1, attr2 in attribute_pairs:
        plt.figure(figsize=(8, 6))

        # Plot points with different colors based on their cluster
        for cluster_id in sorted(y.unique()):
            cluster_points = df_standardized[y == cluster_id]
            plt.scatter(
                cluster_points[attr1],
                cluster_points[attr2],
                label=f"Cluster {cluster_id}",
                alpha=0.7,
            )

        plt.xlabel(attr1)
        plt.ylabel(attr2)
        plt.title(f"{attr1} vs {attr2}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        filename: str = f"cluster_plot_{attr1}_vs_{attr2}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
