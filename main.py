from models import LTSM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import argparse


def main(sequence_length, hidden_dim, iterations, learning_rate, stepper):
    # Importing the training set
    dataset_train = pd.read_csv('train.csv', header=None)
    training_set = dataset_train.values[:, :1]

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 60 timesteps and 1 output;
    x_train = []
    y_train = []
    for i in range(sequence_length + 1, 1258):
        x_train.append(training_set_scaled[i - (sequence_length + 1):i-1, :])
        y_train.append(training_set_scaled[i - sequence_length:i, :])
    x_train, y_train = np.array(x_train), np.array(y_train)

    model = LTSM(x_train.shape[2], sequence_length, hidden_dim=hidden_dim, stepper_enabled=stepper,
                 learning_rate=learning_rate)

    x_train_shuffled, y_train_shuffled = shuffle(x_train, y_train)
    loss, smooth_loss = model.fit(x_train_shuffled, y_train_shuffled, iterations=iterations)
    plt.plot(loss, color="blue")
    plt.plot(smooth_loss, color="orange")
    plt.show()
    dataset_test = pd.read_csv('test.csv', header=None)
    real_stock_price = dataset_test.values[:, :1]
    test_set_scaled = sc.transform(real_stock_price)

    x_test = []
    y_test = []
    test_set = np.concatenate((training_set_scaled, test_set_scaled), axis=0)
    for i in range((sequence_length + 1), 1278):
        x_test.append(test_set[i - (sequence_length + 1):i - 1, :])
        y_test.append(test_set[i - sequence_length:i, :])
    x_test, y_test = np.array(x_test), np.array(y_test)
    predicted = model.predict(x_test)
    predicted = predicted[:, (sequence_length-1), :]
    predicted = sc.inverse_transform(predicted)
    y_test = y_test[:, (sequence_length - 1), :]
    y_test = sc.inverse_transform(y_test)
    open_real = []
    open_predicted = []
    for i in range(x_test.shape[0]):
        open_real.append(y_test[i][0])
        open_predicted.append(predicted[i][0])

    plt.plot(open_real, color="red")
    plt.plot(open_predicted, color="blue")
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LTSM Neural network.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Input sequence length")
    parser.add_argument('--sequence_length', type=int, default=60, help="Input sequence length")
    parser.add_argument('--iterations', type=int, default=1000, help="Count of training iterations")
    parser.add_argument('--hidden_dim', type=int, default=50, help="Dimensions of hidden layers")
    parser.add_argument('--stepper', action="store_true", help="Enable stepper mode")
    args = parser.parse_args()
    main(args.sequence_length, args.hidden_dim, args.iterations, args.learning_rate, args.stepper)
