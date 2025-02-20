from typing import List
import pandas as pd
from adult import *
from pandas.core.series import Series

df: pd.DataFrame = read_csv_1("./data/adult.csv")

num_instances: int = num_rows(df)
attributes: List[str] = column_names(df)
missing_count: int = missing_values(df)
cols_with_missing: List[str] = columns_with_missing_values(df)
bachelors_masters_pct: float = bachelors_masters_percentage(df)

# Print the results
print(f"Number of instances: {num_instances}")
print(f"Attributes: {attributes}")
print(f"Number of missing values: {missing_count}")
print(f"Columns with missing values: {cols_with_missing}")
print(f"Percentage of Bachelors and Masters degrees: {bachelors_masters_pct}%")