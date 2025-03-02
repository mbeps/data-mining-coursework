import os
import pandas as pd
from typing import List

# Import the necessary functions from coronavirus_tweets.py
from coronavirus_tweets import (
    read_csv_3,
    lower_case,
    remove_non_alphabetic_chars,
    remove_multiple_consecutive_whitespaces,
    tokenize,
    count_words_with_repetitions,
    count_words_without_repetitions,
    frequent_words,
    remove_stop_words,
    stemming
)

def print_separator(title: str) -> None:
    """Print a separator with a title for better readability."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "-"))
    print("=" * 80)

def run_tests() -> None:
    """Run tests on the text-mining functions (part 2)."""
    filepath = './data/coronavirus_tweets.csv'
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        print("Please ensure the coronavirus_tweets.csv file is in the data folder.")
        return

    print_separator("TASK 3 PART 2: LOADING & NORMALIZING DATA")

    # 1. Read CSV
    df = read_csv_3(filepath)
    print(f"Dataset loaded with shape: {df.shape}")

    # 2. Normalize tweets (lowercase, remove non-alphabetic, remove extra spaces)
    df = lower_case(df)
    df = remove_non_alphabetic_chars(df)
    df = remove_multiple_consecutive_whitespaces(df)

    # Show some sample tweets after normalization
    print("\nSample normalized tweets:")
    print(df['OriginalTweet'].head(5))
    
    print_separator("TASK 3 PART 2: TOKENIZING TWEETS")

    df = tokenize(df)
    print("First 2 tokenized tweets:")
    print(df['tokenized'].head(2).tolist())
    
    # 3. Count words with repetitions
    total_words = count_words_with_repetitions(df)
    print(f"\nTotal words (with repetitions): {total_words}")

    # 4. Count distinct words
    distinct_count = count_words_without_repetitions(df)
    print(f"Total distinct words: {distinct_count}")

    # 5. 10 most frequent words (before removing stop words)
    top10 = frequent_words(df, 10)
    print(f"\nTop 10 most frequent words (raw): {top10}")
    
    print_separator("TASK 3 PART 2: REMOVING STOP WORDS & SHORT WORDS")

    df = remove_stop_words(df)
    print("First 2 tokenized tweets after removing stop words & short words:")
    print(df['tokenized'].head(2).tolist())
    
    print_separator("TASK 3 PART 2: STEMMING")

    df = stemming(df)
    print("First 2 tokenized tweets after stemming:")
    print(df['tokenized'].head(2).tolist())

    # 6. Recompute 10 most frequent words in the cleaned & stemmed corpus
    top10_cleaned = frequent_words(df, 10)
    print(f"\nTop 10 most frequent words (after cleaning & stemming): {top10_cleaned}")

if __name__ == "__main__":
    run_tests()
