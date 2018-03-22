import pandas as pd

from collections import OrderedDict

from errors import relative_error, mean_relative_error


def calculate_stats(test_y, predicted_y, train_y, predicted_train_y, target_column):
    relative_errors = relative_error(test_y, predicted_y)

    summary_df = pd.DataFrame(OrderedDict((
        (target_column, test_y.ravel()),
        ("Predicted", predicted_y.ravel()),
        ("Relative error", relative_errors.ravel())
    )))

    print(summary_df.head(10))
    print()
    print("Mean relative error for train data:\t{:.4f}".format(mean_relative_error(train_y, predicted_train_y)))
    print("Mean relative error for test data:\t{:.4f}".format(mean_relative_error(test_y, predicted_y)))
