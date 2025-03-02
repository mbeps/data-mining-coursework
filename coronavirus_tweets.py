import pandas as pd
from typing import List
import numpy as np

def read_csv_3(data_file: str) -> pd.DataFrame:
    """
    Return a pandas dataframe containing the dataset.
    Uses 'latin-1' encoding.
    """
    df = pd.read_csv(data_file, encoding='latin-1')
    return df

def get_sentiments(df: pd.DataFrame) -> List[str]:
    """
    Return a list with the possible sentiments that a tweet might have.
    """
    # Use unique to get possible sentiments and convert to list.
    sentiments = df['Sentiment'].dropna().unique().tolist()
    return sentiments

def second_most_popular_sentiment(df: pd.DataFrame) -> str:
    """
    Return a string containing the second most popular sentiment among the tweets.
    """
    # Count the frequency of each sentiment.
    sentiment_counts = df['Sentiment'].value_counts()
    if len(sentiment_counts) < 2:
        # In case there is only one sentiment or none.
        return sentiment_counts.index[0] if len(sentiment_counts) == 1 else ""
    # Return the sentiment with the second highest frequency.
    return sentiment_counts.index[1]

def date_most_popular_tweets(df: pd.DataFrame) -> str:
    """
    Return the date (as it appears in the data) with the greatest number of extremely positive tweets.
    """
    # Filter tweets with sentiment "Extremely Positive"
    extremely_positive = df[df['Sentiment'] == 'Extremely Positive']
    if extremely_positive.empty:
        return ""
    # Count tweets by date (TweetAt column) and return the date with the highest count.
    date_counts = extremely_positive['TweetAt'].value_counts()
    return date_counts.idxmax()

def lower_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the dataframe by converting all tweets to lower case.
    """
    # Assume tweets are in the "OriginalTweet" column.
    df['OriginalTweet'] = df['OriginalTweet'].str.lower()
    return df

def remove_non_alphabetic_chars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the dataframe by replacing every character that is not alphabetical or whitespace with a whitespace.
    """
    df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
    return df

def remove_multiple_consecutive_whitespaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the dataframe by replacing multiple consecutive whitespaces with a single whitespace
    and trimming leading/trailing spaces.
    """
    df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	pass

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	pass

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	pass

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	pass

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	pass

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	pass

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	pass

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	pass





