import os
import pandas as pd
import numpy as np

# Import your functions from coronavirus_tweets.py
from coronavirus_tweets import (
    read_csv_3,
    lower_case,
    remove_non_alphabetic_chars,
    remove_multiple_consecutive_whitespaces,
    mnb_predict,
    mnb_accuracy
)

def print_separator(title: str) -> None:
    """Print a separator with a title for better readability."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "-"))
    print("=" * 80)

def run_tests() -> None:
    """Run tests on the Multinomial Naive Bayes classifier task (part 4)."""
    filepath = './data/coronavirus_tweets.csv'
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        print("Please ensure the coronavirus_tweets.csv file is in the data folder.")
        return
    
    print_separator("TASK 3 PART 4: LOADING & NORMALIZING DATA")
    df = read_csv_3(filepath)
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Optional: normalize tweets for more consistent tokenization
    # (Note: You can also rely on CountVectorizer's built-in tokenization & pre-processing,
    # but normalizing may help if the data is quite messy.)
    df = lower_case(df)
    df = remove_non_alphabetic_chars(df)
    df = remove_multiple_consecutive_whitespaces(df)
    
    print("\nSample of normalized text:")
    print(df['OriginalTweet'].head(5))
    
    print_separator("TASK 3 PART 4: FITTING MULTINOMIAL NAIVE BAYES")
    
    # Predict on the entire training set
    y_pred = mnb_predict(df)
    
    # Measure training accuracy
    y_true = df['Sentiment'].values
    accuracy = mnb_accuracy(y_pred, y_true)
    
    print(f"\nTraining accuracy: {accuracy:.3f}")
    print("Note: This is training accuracy, not a separate test score.")

if __name__ == "__main__":
    run_tests()
