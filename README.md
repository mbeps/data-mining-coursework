## Introduction

This repository contains the programming coursework for the 7CCSMDM1 Data Mining module at King's College London. The assignment focuses on applying classification and cluster analysis techniques to various datasets, requiring efficient data manipulation, model building, and insight extraction. The coursework demonstrates practical experience with fundamental data mining tasks by implementing solutions for three distinct problems using different datasets.

The project is implemented in Python, following the specific guidelines provided in the assignment description. All code is structured in modular functions that meet the input/output specifications required for automated assessment. The coursework is worth 20% of the overall module mark and is assessed out of 100 points.

## Objectives

This coursework tests knowledge and practical skills in the following areas:

### Part 1: Decision Trees with Categorical Attributes (30 points)
- Load and explore the adult dataset from UCI Machine Learning Repository
- Handle missing values and perform feature engineering (one-hot encoding)
- Implement a decision tree classifier to predict income levels
- Evaluate model performance using error rate calculations

### Part 2: Cluster Analysis (30 points)
- Analyze the wholesale customers dataset
- Implement and compare k-means and agglomerative hierarchical clustering
- Standardize data and evaluate cluster quality using Silhouette scores
- Visualize clusters using scatter plots for attribute pairs

### Part 3: Text Mining (40 points)
- Process and analyze the Coronavirus Tweets NLP dataset
- Implement text preprocessing techniques (tokenization, stemming, etc.)
- Extract insights about sentiment distribution
- Build a Multinomial Naïve Bayes classifier for sentiment prediction
- Optimize classification accuracy using vectorization techniques

## Requirements

To run this project, you'll need:

- Python 3.8 or above
- Poetry or Anaconda

## Usage

### Poetry Setup

1. Ensure you have Poetry installed. If not, install it following the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

2. Clone this repository:
   ```bash
   git clone https://github.com/mbeps/data-mining-coursework.git
   cd data-mining-coursework
   ```

3. Create a Poetry environment using the provided configuration:
   ```bash
   poetry install
   ```

4. Activate the Poetry shell:
   ```bash
   poetry shell
   ```

5. Download required NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

6. Run the test scripts to verify the implementations:
   ```bash
   python test_task1.py
   python test_task2.py
   python test_task3.py
   ```

### Anaconda Setup

1. Ensure you have Anaconda or Miniconda installed. If not, install it from the [Anaconda website](https://www.anaconda.com/products/distribution).

2. Clone this repository:
   ```bash
   git clone https://github.com/mbeps/data-mining-coursework.git
   cd data-mining-coursework
   ```

3. Create a conda environment using the provided YAML file:
   ```bash
   conda env create -f dm1_2025.yml
   ```

4. Activate the conda environment:
   ```bash
   conda activate dm1_2024
   ```
   
   Note: Despite the environment name being `dm1_2024` in the file, this is the correct environment for the 2025 coursework.

5. Download required NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

6. Run the test scripts to verify the implementations:
   ```bash
   python test_task1.py
   python test_task2.py
   python test_task3.py
   ```

For detailed information about each task and implementation requirements, refer to the coursework specification document. All datasets should be placed in a `./data` directory relative to the project root.
