import os
import pandas as pd
from typing import List

# Import the functions implemented in the coronavirus_tweets.py module.
from coronavirus_tweets import (
    read_csv_3,
    get_sentiments,
    second_most_popular_sentiment,
    date_most_popular_tweets,
    lower_case,
    remove_non_alphabetic_chars,
    remove_multiple_consecutive_whitespaces,
)

def print_separator(title: str) -> None:
    """Print a separator with a title for better readability."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "-"))
    print("=" * 80)

def run_tests() -> None:
    """Run tests on the implemented text mining functions."""
    filepath = './data/coronavirus_tweets.csv'
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        print("Please ensure the coronavirus_tweets.csv file is in the data folder.")
        return

    print_separator("TASK 3 PART 1: READING AND EXPLORING DATA")
    
    # Test read_csv_3
    print("Loading the dataset...")
    df = read_csv_3(filepath)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    print("\nSample data:")
    print(df.head(), "\n")
    
    # Test get_sentiments
    sentiments = get_sentiments(df)
    print(f"Possible sentiments: {sentiments}")
    
    # Test second_most_popular_sentiment
    second_sent = second_most_popular_sentiment(df)
    print(f"Second most popular sentiment: {second_sent}")
    
    # Test date_most_popular_tweets
    popular_date = date_most_popular_tweets(df)
    print(f"Date with most extremely positive tweets: {popular_date}")
    
    print_separator("TASK 3 PART 1: TEXT NORMALIZATION")
    
    # Create a copy of df for each step to prevent cascading changes
    df_lower = lower_case(df.copy())
    print("Tweets after converting to lower case:")
    print(df_lower['OriginalTweet'].head(), "\n")
    
    df_alpha = remove_non_alphabetic_chars(df_lower.copy())
    print("Tweets after removing non-alphabetic characters:")
    print(df_alpha['OriginalTweet'].head(), "\n")
    
    df_clean = remove_multiple_consecutive_whitespaces(df_alpha.copy())
    print("Tweets after collapsing multiple whitespaces:")
    print(df_clean['OriginalTweet'].head())

if __name__ == "__main__":
    run_tests()
