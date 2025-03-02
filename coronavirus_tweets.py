import pandas as pd
import numpy as np
import requests
from typing import List
import nltk

# Make sure you have nltk's PorterStemmer available, e.g.:
# import nltk
# nltk.download('punkt')  # if needed
# nltk.download('stopwords') # if needed

from nltk.stem import PorterStemmer

def read_csv_3(data_file: str) -> pd.DataFrame:
    """
    Return a pandas dataframe containing the data set.
    Uses 'latin-1' encoding.
    """
    df = pd.read_csv(data_file, encoding='latin-1')
    return df

def get_sentiments(df: pd.DataFrame) -> List[str]:
    """
    Return a list with the possible sentiments that a tweet might have.
    """
    sentiments = df['Sentiment'].dropna().unique().tolist()
    return sentiments

def second_most_popular_sentiment(df: pd.DataFrame) -> str:
    """
    Return a string containing the second most popular sentiment among the tweets.
    """
    sentiment_counts = df['Sentiment'].value_counts()
    if len(sentiment_counts) < 2:
        return sentiment_counts.index[0] if len(sentiment_counts) == 1 else ""
    return sentiment_counts.index[1]

def date_most_popular_tweets(df: pd.DataFrame) -> str:
    """
    Return the date (string as it appears in the data) 
    with the greatest number of extremely positive tweets.
    """
    extremely_positive = df[df['Sentiment'] == 'Extremely Positive']
    if extremely_positive.empty:
        return ""
    date_counts = extremely_positive['TweetAt'].value_counts()
    return date_counts.idxmax()

def lower_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the dataframe by converting all tweets to lower case.
    """
    df['OriginalTweet'] = df['OriginalTweet'].str.lower()
    return df

def remove_non_alphabetic_chars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the dataframe by replacing every character that is not 
    alphabetical or whitespace with a whitespace.
    """
    df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
    return df

def remove_multiple_consecutive_whitespaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the dataframe by replacing multiple consecutive whitespaces 
    with a single whitespace and trimming leading/trailing spaces.
    """
    df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df

def tokenize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe where each tweet is one string with words 
    separated by single whitespaces, tokenize every tweet by 
    converting it into a list of words (strings).
    
    Creates a new column 'tokenized' in df with a list of tokens per tweet.
    """
    df['tokenized'] = df['OriginalTweet'].apply(lambda x: x.split() if isinstance(x, str) else [])
    return df

def count_words_with_repetitions(tdf: pd.DataFrame) -> int:
    """
    Given dataframe tdf with the tweets tokenized, 
    return the number of words in all tweets including repetitions.
    """
    # Summation of lengths of each token list
    total_words = tdf['tokenized'].apply(len).sum()
    return total_words

def count_words_without_repetitions(tdf: pd.DataFrame) -> int:
    """
    Given dataframe tdf with the tweets tokenized, 
    return the number of distinct words in all tweets combined.
    """
    all_words = set()
    for tokens in tdf['tokenized']:
        all_words.update(tokens)
    return len(all_words)

def frequent_words(tdf: pd.DataFrame, k: int) -> List[str]:
    """
    Given dataframe tdf with the tweets tokenized, 
    return a list with the k distinct words that are most frequent in the tweets.
    """
    from collections import Counter
    
    # Flatten all tokens
    all_tokens = []
    for tokens in tdf['tokenized']:
        all_tokens.extend(tokens)
        
    freq_counter = Counter(all_tokens)
    # Extract the k most common words
    most_common = freq_counter.most_common(k)
    # Return only the words (without counts)
    return [word for word, _ in most_common]

def remove_stop_words(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    Given dataframe tdf with the tweets tokenized, remove stop words and 
    words with <= 2 characters from each tweet.
    
    The function should download the list of stop words via:
    https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
    """
    # Download stop words
    url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
    response = requests.get(url)
    stopwords_str = response.text
    stopwords = set(stopwords_str.split())
    
    def filter_tokens(tokens):
        return [
            w for w in tokens 
            if w not in stopwords and len(w) > 2
        ]
    
    tdf['tokenized'] = tdf['tokenized'].apply(filter_tokens)
    return tdf

def stemming(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    Given dataframe tdf with the tweets tokenized, reduce each word 
    in every tweet to its stem (via e.g. PorterStemmer).
    """
    stemmer = PorterStemmer()
    
    def stem_tokens(tokens):
        return [stemmer.stem(t) for t in tokens]
    
    tdf['tokenized'] = tdf['tokenized'].apply(stem_tokens)
    return tdf

def mnb_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
    build a Multinomial Naive Bayes classifier. 
    Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
    as a 1d array (numpy.ndarray).
    
    [Placeholder: To be completed in a later step if desired.]
    """
    return np.array([])

def mnb_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Given a 1d array (numpy.ndarray) y_pred with predicted labels 
    (e.g. 'Neutral', 'Positive') by a classifier and another 1d array y_true with 
    the true labels, return the classification accuracy rounded to the 3rd decimal digit.
    
    [Placeholder: To be completed in a later step if desired.]
    """
    return 0.0
