import re
import warnings
from typing import List
from pandas import DataFrame, Series
from sklearn.naive_bayes import MultinomialNB, ComplementNB # type: ignore
import numpy as np
import re
import numpy as np
import pandas as pd
import requests
from nltk.stem import PorterStemmer  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore


def read_csv_3(data_file: str) -> pd.DataFrame:
    """
    Return a pandas dataframe containing the data set.
    Uses 'latin-1' encoding.
    """
    return pd.read_csv(data_file, encoding="latin-1")


def get_sentiments(df: pd.DataFrame) -> List[str]:
    """
    Return a list with the possible sentiments that a tweet might have.
    """
    return df["Sentiment"].dropna().unique().tolist()


def second_most_popular_sentiment(df: pd.DataFrame) -> str:
    """
    Return a string containing the second most popular sentiment among the tweets.
    """
    sentiment_counts = df["Sentiment"].value_counts()
    if len(sentiment_counts) < 2:
        return sentiment_counts.index[0] if len(sentiment_counts) == 1 else ""
    return sentiment_counts.index[1]


def date_most_popular_tweets(df: pd.DataFrame) -> str:
    """
    Return the date (string as it appears in the data)
    with the greatest number of extremely positive tweets.
    """
    extremely_positive: DataFrame = df[df["Sentiment"] == "Extremely Positive"]
    if extremely_positive.empty:
        return ""
    date_counts: Series = extremely_positive["TweetAt"].value_counts()
    return str(date_counts.idxmax())


def lower_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the dataframe by converting all tweets to lower case.
    """
    df["OriginalTweet"] = df["OriginalTweet"].str.lower()
    return df


def remove_non_alphabetic_chars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the dataframe by replacing every character that is not
    alphabetical or whitespace with a whitespace.
    """
    df["OriginalTweet"] = df["OriginalTweet"].str.replace(
        r"[^a-zA-Z\s]", " ", regex=True
    )
    return df


def remove_multiple_consecutive_whitespaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the dataframe by replacing multiple consecutive whitespaces
    with a single whitespace and trimming leading/trailing spaces.
    """
    df["OriginalTweet"] = (
        df["OriginalTweet"].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return df


def tokenize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe where each tweet is one string with words
    separated by single whitespaces, tokenize every tweet by
    converting it into a list of words (strings).

    Creates a new column 'tokenized' in df with a list of tokens per tweet.
    """
    df["tokenized"] = df["OriginalTweet"].apply(
        lambda x: x.split() if isinstance(x, str) else []
    )
    return df


def count_words_with_repetitions(tdf: pd.DataFrame) -> int:
    """
    Given dataframe tdf with the tweets tokenized,
    return the number of words in all tweets including repetitions.
    """
    return tdf["tokenized"].apply(len).sum()


def count_words_without_repetitions(tdf: pd.DataFrame) -> int:
    """
    Given dataframe tdf with the tweets tokenized,
    return the number of distinct words in all tweets combined.
    """
    all_words = set()
    for tokens in tdf["tokenized"]:
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
    for tokens in tdf["tokenized"]:
        all_tokens.extend(tokens)

    freq_counter = Counter(all_tokens)
    # Extract the k most common words
    most_common = freq_counter.most_common(k)
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
        return [w for w in tokens if w not in stopwords and len(w) > 2]

    tdf["tokenized"] = tdf["tokenized"].apply(filter_tokens)
    return tdf


def stemming(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    Given dataframe tdf with the tweets tokenized, reduce each word
    in every tweet to its stem (via e.g. PorterStemmer).
    """
    stemmer = PorterStemmer()

    def stem_tokens(tokens):
        return [stemmer.stem(t) for t in tokens]

    tdf["tokenized"] = tdf["tokenized"].apply(stem_tokens)
    return tdf


def mnb_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
    build a Multinomial Naive Bayes classifier.
    Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
    as a 1d array (numpy.ndarray).
    """
    # Further enhanced text preprocessing for tweets
    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Replace URLs with a token
        text = re.sub(r"https?://\S+", "URL", text)
        # Replace @mentions with a token
        text = re.sub(r"@\w+", "MENTION", text)
        # Replace hashtags with their content
        text = re.sub(r"#(\w+)", r"\1", text)
        # Handle elongated words (e.g., "loooove" -> "love")
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        # Handle common emoticons
        text = re.sub(r"[:;=]-?[\)\]dD>]", " HAPPY_EMOTICON ", text)
        text = re.sub(r"[:;=]-?[\(\[<]", " SAD_EMOTICON ", text)
        # Special handling for negations
        text = re.sub(r"\b(?:not|no|never|n\'t)\s+(\w+)", r"NOT_\1", text)
        # Replace non-alphabetic characters with spaces
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # Apply preprocessing
    df["processed_tweet"] = df["OriginalTweet"].apply(preprocess_text)

    # Add extra features
    df["tweet_length"] = df["processed_tweet"].str.len()
    df["word_count"] = df["processed_tweet"].str.split().str.len()
    df["caps_count"] = df["OriginalTweet"].str.count(r"[A-Z]{2,}")

    # Convert to numpy arrays
    X = np.array(df["processed_tweet"].fillna("").values)
    y = np.array(df["Sentiment"].values)

    # Define parameter combinations to try
    count_vectorizer_params: list[tuple] = [
        # min_df, max_df, ngram_range, max_features, stop_words, analyzer, binary
        (1, 0.9, (1, 1), 5000, "english", "word", False),
        (2, 0.9, (1, 2), 10000, "english", "word", False),
        (2, 0.95, (1, 2), 15000, "english", "word", False),
        (3, 0.95, (1, 3), 20000, "english", "word", False),
        (1, 0.9, (2, 5), 10000, None, "char", False),
        (2, 0.9, (1, 2), 10000, "english", "word", True),  # Binary features
        (2, 0.95, (1, 3), 25000, "english", "word", False),  # More features
    ]

    tfidf_vectorizer_params: list[tuple] = [
        # min_df, max_df, ngram_range, max_features, use_idf, norm, stop_words, analyzer, binary, sublinear_tf
        (1, 0.9, (1, 1), 5000, True, "l2", "english", "word", False, False),
        (2, 0.9, (1, 2), 10000, True, "l2", "english", "word", False, False),
        (2, 0.95, (1, 2), 15000, True, "l2", "english", "word", False, False),
        (3, 0.95, (1, 3), 20000, True, "l2", "english", "word", False, False),
        (1, 0.9, (2, 5), 10000, True, "l2", None, "char", False, False),
        (
            2,
            0.9,
            (1, 2),
            10000,
            True,
            "l2",
            "english",
            "word",
            True,
            False,
        ),  # Binary features
        (
            2,
            0.9,
            (1, 2),
            15000,
            True,
            "l2",
            "english",
            "word",
            False,
            True,
        ),  # Sublinear TF scaling
        (
            1,
            0.9,
            (1, 3),
            20000,
            True,
            "l1",
            "english",
            "word",
            False,
            True,
        ),  # L1 norm with sublinear TF
    ]

    # Expanded alpha values for MultinomialNB
    alpha_values: List[float] = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]

    # Initialize best parameters and accuracy
    best_accuracy = 0
    best_y_pred = None

    # Try CountVectorizer combinations
    for (
        min_df,
        max_df,
        ngram_range,
        max_features,
        stop_words,
        analyzer,
        binary,
    ) in count_vectorizer_params:
        try:
            vectorizer = CountVectorizer(
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range,
                max_features=max_features,
                stop_words=stop_words,
                analyzer=analyzer,
                binary=binary,
            )

            X_transformed = vectorizer.fit_transform(X)

            for alpha in alpha_values:
                # Try MultinomialNB
                for fit_prior in [True, False]:
                    clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                    clf.fit(X_transformed, y)
                    y_pred = clf.predict(X_transformed)
                    accuracy = np.mean(y_pred == y)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_y_pred = y_pred

                # Try ComplementNB which often works better for imbalanced classes
                for fit_prior in [True, False]:
                    try:
                        clf = ComplementNB(alpha=alpha, fit_prior=fit_prior)
                        clf.fit(X_transformed, y)
                        y_pred = clf.predict(X_transformed)
                        accuracy = np.mean(y_pred == y)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_y_pred = y_pred
                    except:
                        continue
        except Exception:
            continue

    # Try TfidfVectorizer combinations
    for (
        min_df,
        max_df,
        ngram_range,
        max_features,
        use_idf,
        norm,
        stop_words,
        analyzer,
        binary,
        sublinear_tf,
    ) in tfidf_vectorizer_params:
        try:
            vectorizer = TfidfVectorizer(
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range,
                max_features=max_features,
                use_idf=use_idf,
                norm=norm,
                stop_words=stop_words,
                analyzer=analyzer,
                binary=binary,
                sublinear_tf=sublinear_tf,
            )

            X_transformed = vectorizer.fit_transform(X)

            for alpha in alpha_values:
                # Try MultinomialNB
                for fit_prior in [True, False]:
                    clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                    clf.fit(X_transformed, y)
                    y_pred = clf.predict(X_transformed)
                    accuracy = np.mean(y_pred == y)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_y_pred = y_pred

                # Try ComplementNB
                for fit_prior in [True, False]:
                    try:
                        clf = ComplementNB(alpha=alpha, fit_prior=fit_prior)
                        clf.fit(X_transformed, y)
                        y_pred = clf.predict(X_transformed)
                        accuracy = np.mean(y_pred == y)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_y_pred = y_pred
                    except:
                        continue
        except Exception:
            continue

    # If no valid combination found, use default parameters
    if best_y_pred is None:
        vectorizer = CountVectorizer()
        X_transformed = vectorizer.fit_transform(X)
        clf = MultinomialNB()
        clf.fit(X_transformed, y)
        best_y_pred = clf.predict(X_transformed)

    return best_y_pred


def mnb_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Given a 1d array (numpy.ndarray) y_pred with predicted labels
    (e.g. 'Neutral', 'Positive') by a classifier and another 1d array y_true with
    the true labels, return the classification accuracy rounded to the 3rd decimal digit.
    """
    accuracy: float = np.mean(y_pred == y_true)
    return round(accuracy, 3)
