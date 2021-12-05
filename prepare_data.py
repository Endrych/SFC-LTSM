import pandas as pd

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

dataset_train = pd.read_csv('Google_Stock_Price_Test.csv')
training_set = dataset_train.iloc[:, 1:5].values
training_set_shape = training_set.shape
for x in range(training_set_shape[0]):
    for y in range(training_set_shape[1]):
        if isinstance(training_set[x][y], str):
            training_set[x][y] = locale.atof(training_set[x][y])
pd.DataFrame(training_set).to_csv('test.csv', header=False, index=False)
