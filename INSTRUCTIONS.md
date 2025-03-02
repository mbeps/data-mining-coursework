# 7CCSMDM1 Data Mining

## Programming Coursework

**Due:** Thursday 20 March 2025 (4pm UK time)

The aim of this coursework assignment is to demonstrate understanding of and obtain experience with classification and cluster analysis, which are among the most important data mining tasks. It requires efficiently manipulating data sets, building data mining models, and obtaining insights with different types of data. 

The coursework is worth **20%** of the overall module mark and will be marked out of **100 points**. The distribution of points is:

- **30 points** for decision trees with categorical attributes.
- **30 points** for cluster analysis.
- **40 points** for text mining.

The data sets required for this coursework are provided on the KEATS page of the module. Do not download them from their original sources for your coursework implementation. The links to their origins are provided for referencing purposes. Instructions are given at the end of this document. 

It is important that you carefully follow all coursework instructions because the coursework will be automatically marked.

---

## 1. Decision Trees with Categorical Attributes

This part uses the [adult data set](https://archive.ics.uci.edu/ml/datasets/Adult) from the UCI Machine Learning Repository to predict whether the income of an individual exceeds 50K per year based on 14 attributes. The attribute `fnlwgt` should be dropped, and the following attributes should be taken into consideration:

| Attribute      | Description                   |
| -------------- | ----------------------------- |
| age            | Age group                     |
| workclass      | Type of employment            |
| education      | Level of education reached    |
| education-num  | Number of education years     |
| marital-status | Type of marital status        |
| occupation     | Occupation domain             |
| relationship   | Type of relationship involved |
| race           | Social category               |
| sex            | Male or female                |
| capital-gain   | Class of capital gains        |
| capital-loss   | Class of capital losses       |
| hours-per-week | Category of working hours     |
| native-country | Country of birth              |

### Tasks

1. **(10 points)** Load the data set and compute:
   - (a) The number of instances.
   - (b) A list with the attribute names.
   - (c) The number of missing attribute values.
   - (d) A list of the attribute names with at least one missing value.
   - (e) The percentage of instances corresponding to individuals whose education level is Bachelors or Masters.

2. **(10 points)** Drop all instances with missing values. Convert all input attributes to numeric using one-hot encoding. Name the new columns using attribute values from the original data set. Next, convert the class values to numeric with label encoding.

3. **(10 points)** Build a decision tree and classify each instance into one of the `<= 50K` and `> 50K` categories. Compute the training error rate of the resulting tree.

4. *(Optional, Not marked)* After completing the above, optionally perform **N-fold cross-validation** for different values of N. What is a reasonable choice for N?

5. *(Optional, Not marked)* Evaluate the effect of erroneous values in the training data. Perturb a portion **p%** of the attribute values in the training set and assess the impact on classification accuracy.

---

## 2. Cluster Analysis

This part uses the [wholesale customers data set](https://archive.ics.uci.edu/ml/datasets/wholesale+customers) from the UCI Machine Learning Repository to identify similar groups of customers based on 8 attributes. The attributes **Channel** and **Region** should be dropped, and only the following 6 numeric attributes should be considered:

| Attribute    | Description                              |
| ------------ | ---------------------------------------- |
| Fresh        | Annual expenses on fresh products        |
| Milk         | Annual expenses on milk products         |
| Grocery      | Annual expenses on grocery products      |
| Frozen       | Annual expenses on frozen products       |
| Detergent    | Annual expenses on detergent products    |
| Delicatessen | Annual expenses on delicatessen products |

### Tasks

1. **(10 points)** Compute the mean, standard deviation, minimum, and maximum value for each attribute. Round the mean and standard deviation to the closest integers.

2. **(20 points)** Divide the data points into k clusters, for \( k \in \{3, 5, 10\} \), using k-means and agglomerative hierarchical clustering. Repeat k-means **10 times** for each \( k \) value. Next, standardize each attribute value and repeat the clustering. Evaluate using the **Silhouette score**. 

3. *(Optional, Not marked)* Evaluate how the quality of the resulting clusters is affected if **k-means++** is used instead of standard k-means.

---

## 3. Text Mining

This part uses the [Coronavirus Tweets NLP data set](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification) from Kaggle to predict the sentiment of tweets related to COVID-19. The dataset contains 6 attributes:

| Attribute     | Description                     |
| ------------- | ------------------------------- |
| UserName      | Anonymised attribute            |
| ScreenName    | Anonymised attribute            |
| Location      | Location of the person tweeting |
| TweetAt       | Date                            |
| OriginalTweet | Text content of the tweet       |
| Sentiment     | Emotion of the tweet            |

### Tasks

1. **(13 points)** Compute:
   - The possible sentiments a tweet may have.
   - The second most popular sentiment.
   - The date with the most extremely positive tweets.
   - Convert tweets to lower case, replace non-alphabetical characters with whitespace, and ensure words are separated by a single whitespace.

2. **(14 points)** Tokenize tweets and count:
   - Total words (including repetitions).
   - Total distinct words.
   - 10 most frequent words before and after removing stop words and stemming.

3. *(Optional, Not marked)* Plot a word frequency distribution. Use a line chart to represent the fraction of documents in which a word appears.

4. **(13 points)** Store the dataset in a numpy array and create a sparse term-document matrix using `CountVectorizer`. Train a **Multinomial Naive Bayes classifier** and tune parameters for best accuracy.

---

## Instructions

- Use the provided template files: `adult.py`, `wholesale_customers.py`, and `coronavirus_tweets.py`.
- Do **not** modify template filenames or function names.
- Do **not** convert files to another format (e.g., `.ipynb`).
- Do **not** add code outside function bodies except for import statements.
- Your submission should be a **zipped folder** containing only the three Python files.
- Name your submission: `firstname_lastname_studentnumber.zip`.
- Do **not** include additional files (e.g., CSV files or plots).
- Use **pandas** for data manipulation and **scikit-learn** for model building.
- Implement the coursework **individually**.

---

**Topics Covered:**
- Decision Trees with Categorical Attributes
- Cluster Analysis
- Text Mining
