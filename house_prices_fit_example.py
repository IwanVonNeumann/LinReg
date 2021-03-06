import os
import pandas as pd
from sklearn.linear_model import LinearRegression as ScikitLinearRegression

from houses_plots import plot_by_dimensions
from houses_stats import calculate_stats
from lin_regression import LinearRegression

data_dir = "data"
data_file_name = "house_prices.csv"
data_file_path = os.path.join(data_dir, data_file_name)

houses_df = pd.read_csv(data_file_path)

predictor_columns = ["OverallQual", "OverallCond", "GrLivArea", "GarageCars", "YearBuilt", "TotRmsAbvGrd"]
target_column = "SalePrice"

n_records = houses_df.shape[0]
n_train = int(n_records * 0.9)
n_test = int(n_records * 0.1)

train_X = houses_df[predictor_columns].iloc[:n_train, :].as_matrix()
train_y = houses_df[[target_column]].iloc[:n_train, :].as_matrix()

test_X = houses_df[predictor_columns].iloc[-n_test:, :].as_matrix()
test_y = houses_df[[target_column]].iloc[-n_test:, :].as_matrix()

print("{} records split into {} for training and {} for testing\n".format(n_records, n_train, n_test))

lin_reg = LinearRegression()
# lin_reg = ScikitLinearRegression()

lin_reg.fit(train_X, train_y)
predicted_y = lin_reg.predict(test_X)
predicted_train_y = lin_reg.predict(train_X)

calculate_stats(test_y, predicted_y, train_y, predicted_train_y, target_column)
# plot_by_dimensions(lin_reg, test_X, test_y, predicted_y, predictor_columns, target_column)
