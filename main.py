from models import LTSM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

sequence_length = 60

def main():
    # Importing the training set
    dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
    training_set = dataset_train.iloc[:, 1:5].values
    training_set_shape = training_set.shape
    for x in range(training_set_shape[0]):
        for y in range(training_set_shape[1]):
            if isinstance(training_set[x][y], str):
                training_set[x][y] = locale.atof(training_set[x][y])

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 60 timesteps and 1 output;
    X_train = []
    y_train = []
    for i in range(sequence_length + 1, 1258):
        X_train.append(training_set_scaled[i - (sequence_length + 1):i-1, :])
        y_train.append(training_set_scaled[i - sequence_length:i, :])
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = LTSM(X_train.shape[2], X_train.shape[1], hidden_dim=4)
    # model.load('50.mod')
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)
    loss, smooth_loss = model.fit(X_train_shuffled, y_train_shuffled, iterations=2000)
    plt.plot(loss, color="blue")
    plt.plot(smooth_loss, color="orange")
    plt.show()
    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:5].values
    test_set_shape = real_stock_price.shape
    for x in range(test_set_shape[0]):
        for y in range(test_set_shape[1]):
            if isinstance(real_stock_price[x][y], str):
                real_stock_price[x][y] = locale.atof(real_stock_price[x][y])
    test_set_scaled = sc.transform(real_stock_price)

    X_test = []
    y_test = []
    test_set = np.concatenate((training_set_scaled, test_set_scaled), axis=0)
    for i in range(61, 1278):
        X_test.append(test_set[i - (sequence_length + 1):i - 1, :])
        y_test.append(test_set[i - sequence_length:i, :])
    X_test, y_test = np.array(X_test), np.array(y_test)
    predicted = model.predict(X_test)
    predicted = predicted[:, sequence_length-1, :]
    predicted = sc.inverse_transform(predicted)
    y_test = y_test[:, sequence_length-1, :]
    y_test = sc.inverse_transform(y_test)
    open_real = []
    open_predicted = []
    for i in range(X_test.shape[0]):
        open_real.append(y_test[i][0])
        open_predicted.append(predicted[i][0])

    plt.plot(open_real, color="red")
    plt.plot(open_predicted, color="blue")
    plt.show()
    # model.save('50.mod')
    return


if __name__ == "__main__":
    main()
