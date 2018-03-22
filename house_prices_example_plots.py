import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lin_regressor import LinRegressor

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


def plot_dimension(plot_df, dimension, subplot):
    dimension_df = plot_df[[dimension, target_column, "Predicted"]].sort_values(by=[dimension])

    line_X = [dimension_df[dimension].iloc[0], dimension_df[dimension].iloc[-1]]
    line_y = [dimension_df["Predicted"].iloc[0], dimension_df["Predicted"].iloc[-1]]

    subplot.plot(line_X, line_y, linewidth=2.0)
    subplot.plot(dimension_df[dimension], dimension_df[target_column], 'bo')
    subplot.title("{} / price".format(dimension))


plot_data = np.concatenate((test_X, test_y, predicted_y), axis=1)
plot_df_columns = predictor_columns + [target_column, "Predicted"]
plot_df = pd.DataFrame(data=plot_data, columns=plot_df_columns).sample(40, random_state=0)

for i, dimension in enumerate(predictor_columns):
    plt.subplot(2, 3, i + 1)
    plot_dimension(plot_df, dimension, plt)

plt.show()
