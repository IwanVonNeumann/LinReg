import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lin_regressor import LinRegressor

data_dir = "data"
data_file_name = "houses.csv"
data_file_path = os.path.join(data_dir, data_file_name)

houses_df = pd.read_csv(data_file_path)

# print(houses_df.shape)

# print(houses_df.head())

selected_columns = ["OverallQual", "OverallCond", "GrLivArea", "GarageCars", "YearBuilt", "TotRmsAbvGrd"]
target_column = "SalePrice"

houses_df = houses_df[selected_columns + [target_column]].copy()

print(houses_df.head())

n_records = houses_df.shape[0]
n_train = int(n_records * 0.9)
n_test = int(n_records * 0.1)

# print(n_records, n_train, n_test)

train_df = houses_df[:n_train]
test_df = houses_df[-n_test:]
