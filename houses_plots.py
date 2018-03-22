import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_by_dimensions(test_X, test_y, predicted_y, predictor_columns, target_column):
    plot_data = np.concatenate((test_X, test_y, predicted_y), axis=1)
    plot_df_columns = predictor_columns + [target_column, "Predicted"]
    plot_df = pd.DataFrame(data=plot_data, columns=plot_df_columns).sample(40, random_state=0)

    for i, dimension in enumerate(predictor_columns):
        plt.subplot(2, 3, i + 1)
        plot_dimension(plot_df, dimension, target_column, plt)

    plt.show()


def plot_dimension(plot_df, dimension, target_column, subplot):
    dimension_df = plot_df[[dimension, target_column, "Predicted"]].sort_values(by=[dimension])

    line_X = [dimension_df[dimension].iloc[0], dimension_df[dimension].iloc[-1]]
    line_y = [dimension_df["Predicted"].iloc[0], dimension_df["Predicted"].iloc[-1]]

    subplot.plot(line_X, line_y, linewidth=2.0)
    subplot.plot(dimension_df[dimension], dimension_df[target_column], 'bo')
    subplot.title("{} / price".format(dimension))
