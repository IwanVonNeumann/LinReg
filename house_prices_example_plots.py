import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict
from lin_regressor import LinRegressor
from errors import relative_error, mean_relative_error

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

linReg = LinRegressor()
linReg.fit(train_X, train_y)
predicted_y = linReg.predict(test_X)
predicted_train_y = linReg.predict(train_X)

relative_errors = relative_error(test_y, predicted_y)

summary_df = pd.DataFrame(OrderedDict((
    (target_column, test_y.ravel()),
    ("Predicted", predicted_y.ravel()),
    ("Relative error", relative_errors.ravel())
)))

# TODO fix!!!
plot_X = test_X[:10, 0].ravel()  # OverallQual
plot_y = test_y[:10, :].ravel()

plot_X.sort()
plot_y.sort()

print(plot_X)
print(plot_y)

y_ = linReg.predict(test_X[:10])

line_X = [plot_X[0], plot_X[-1]]
line_y = [y_[0], y_[-1]]

plt.plot(line_X, line_y, linewidth=2.0)
plt.plot(plot_X, plot_y, 'bo')
# plt.axis([0, n + 1, 0, f(n + 1) + 1])
plt.show()
