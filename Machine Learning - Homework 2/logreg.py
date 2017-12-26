import random
import argparse
import gzip
import pickle

import numpy as np
from math import exp, log
from random import shuffle
from collections import defaultdict
import matplotlib.pyplot as plt
from copy import deepcopy

SEED = 1735

random.seed(SEED)


class Numbers:
    """
    Class to store MNIST data for images of 0 and 1 only
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if you'd like

        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        train_indices = np.where(self.train_y > 7)
        self.train_x, self.train_y = self.train_x[train_indices], self.train_y[train_indices]
        self.train_y = self.train_y - 8

        self.valid_x, self.valid_y = valid_set
        valid_indices = np.where(self.valid_y > 7)
        self.valid_x, self.valid_y = self.valid_x[valid_indices], self.valid_y[valid_indices]
        self.valid_y = self.valid_y - 8

        self.test_x, self.test_y = test_set
        test_indices = np.where(self.test_y > 7)
        self.test_x, self.test_y = self.test_x[test_indices], self.test_y[test_indices]
        self.test_y = self.test_y - 8

    @staticmethod
    def shuffle(X, y):
        """ Shuffle training data """
        shuffled_indices = np.random.permutation(len(y))
        return X[shuffled_indices], y[shuffled_indices]


class LogReg:
    def __init__(self, num_features, eta):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param eta: A function that takes the iteration as an argument (the default is a constant value)
        """

        self.w = np.zeros(num_features)
        self.eta = eta
        self.last_update = defaultdict(int)

    def progress(self, examples_x, examples_y):
        """
        Given a set of examples, compute the probability and accuracy
        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for x_i, y in zip(examples_x, examples_y):
            p = sigmoid(self.w.dot(x_i))
            if y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples_y))

    def sgd_update(self, x_i, y):
        """
        Compute a stochastic gradient update to improve the log likelihood.
        :param x_i: The features of the example to take the gradient with respect to
        :param y: The target output of the example to take the gradient with respect to
        :return: Return the new value of the regression coefficients
        """

        # TODO: Finish this function to do a single stochastic gradient descent update
        # and return the updated weight vector

        # SOLUTION
        pi = sigmoid(np.dot(self.w, x_i))
        for jj in range(0,x_i.size):
            dl_dbeta =  - 1*(y-pi)*x_i[jj]
            self.w[jj] = self.w[jj] - (self.eta)*dl_dbeta

        return self.w

def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.
    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * np.sign(score)

    # TODO: Finish this function to return the output of applying the sigmoid
    # function to the input score (Please do not use external libraries)

    # SOLUTION:
    return 1/(1+exp(-1*score))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eta", help="Initial SGD learning rate",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--passes", help="Number of passes through training data",
                           type=int, default=1, required=False)

    args = argparser.parse_args()

    data = Numbers("/Users/wilder/PycharmProjects/Machine_learning_HW2/mnist.pkl.gz")

    # Initialize model
    lr = LogReg(data.train_x.shape[1], args.eta)

    # Iterations
    iteration = 0

    for epoch in range(args.passes):

        data.train_x, data.train_y = Numbers.shuffle(data.train_x, data.train_y)

        # TODO: Finish the code to loop over the training data and perform a stochastic
        # gradient descent update on each training example.

        # NOTE: It may be helpful to call upon the 'progress' method in the LogReg
        # class to make sure the algorithm is truly learning properly on both training and test data

        # SOLUTION
        y = data.train_y
        x_train = data.train_x
        for ii in range(0,x_train.shape[0]):
            lr.sgd_update(x_train[ii,:],y[ii])

        print(lr.progress(data.train_x, data.train_y))


    plot_1 = 0
    if plot_1:
        etas = [.0000001, .0001, .01, .1, 1]
        for eta in etas:
            # Initialize model
            lr = LogReg(data.train_x.shape[1], eta)

            # Iterations
            iteration = 0

            # store values
            storage = []
            storage.append(lr.progress(data.train_x, data.train_y)[1])

            epochs = [ii for ii in range(0,10)]
            for epoch in epochs:

                data.train_x, data.train_y = Numbers.shuffle(data.train_x, data.train_y)

                # TODO: Finish the code to loop over the training data and perform a stochastic
                # gradient descent update on each training example.

                # NOTE: It may be helpful to call upon the 'progress' method in the LogReg
                # class to make sure the algorithm is truly learning properly on both training and test data

                # SOLUTION
                y = data.train_y
                x_train = data.train_x
                for ii in range(0, x_train.shape[0]):
                    lr.sgd_update(x_train[ii, :], y[ii])

                storage.append(lr.progress(data.train_x, data.train_y)[1])

                print(epoch,eta)

            new_val = deepcopy(storage)
            plt.plot(new_val, label=str(eta))
        plt.title("Change in training error over multiple epochs")
        plt.xlabel("epochs")
        plt.ylabel("Training accuracy")
        plt.legend([str(ii) for ii in etas])
        plt.show()

    plot_2 = 0
    if plot_2:
        etas = [.0000001, .000001, .00001, .0001, .001, .01, .1, 1]
        for eta in etas:
            # Initialize model
            lr = LogReg(data.train_x.shape[1], eta)

            # Iterations
            iteration = 0

            # store values
            storage = []
            epochs = [1]
            for epoch in epochs:

                data.train_x, data.train_y = Numbers.shuffle(data.train_x, data.train_y)

                # TODO: Finish the code to loop over the training data and perform a stochastic
                # gradient descent update on each training example.

                # NOTE: It may be helpful to call upon the 'progress' method in the LogReg
                # class to make sure the algorithm is truly learning properly on both training and test data

                # SOLUTION
                y = data.train_y
                x_train = data.train_x
                for ii in range(0, 1000):
                    lr.sgd_update(x_train[ii, :], y[ii])
                    storage.append(lr.progress(data.train_x, data.train_y)[0])
                print(storage)

            new_val = deepcopy(storage)
            plt.plot(new_val, label = str(eta))
        plt.title("Change in loss function over single epoch")
        plt.xlabel("Iterations of SGD")
        plt.ylabel("Loss function value")
        plt.legend([str(ii) for ii in etas])
        plt.show()