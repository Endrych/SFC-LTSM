from models import LTSM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import argparse


def main(sequence_length, hidden_dim, iterations, debug):
    # Importing the training set
    dataset_train = pd.read_csv('train.csv', header=None)
    training_set = dataset_train.values

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

    model = LTSM(X_train.shape[2], sequence_length, hidden_dim=hidden_dim, debug=debug)
    # model.load('50.mod')
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)
    loss, smooth_loss = model.fit(X_train_shuffled, y_train_shuffled, iterations=iterations)
    plt.plot(loss, color="blue")
    plt.plot(smooth_loss, color="orange")
    plt.show()
    dataset_test = pd.read_csv('test.csv', header=None)
    real_stock_price = dataset_test.values
    test_set_scaled = sc.transform(real_stock_price)

    X_test = []
    y_test = []
    test_set = np.concatenate((training_set_scaled, test_set_scaled), axis=0)
    for i in range((sequence_length + 1), 1278):
        X_test.append(test_set[i - (sequence_length + 1):i - 1, :])
        y_test.append(test_set[i - sequence_length:i, :])
    X_test, y_test = np.array(X_test), np.array(y_test)
    predicted = model.predict(X_test)
    predicted = predicted[:, (sequence_length-1), :]
    predicted = sc.inverse_transform(predicted)
    y_test = y_test[:, (sequence_length - 1), :]
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
    parser = argparse.ArgumentParser(description='LTSM Neural network.')
    parser.add_argument('--sequence_length', type=int, default=60)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=4)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    main(args.sequence_length, args.hidden_dim, args.iterations, args.debug)
