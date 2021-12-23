from q2.utils import sigmoid

import numpy as np
import math


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    n = data.shape[0]
    ones = np.ones((n,1))
    Xnew = np.hstack((data,ones))
    y = sigmoid(np.dot(Xnew, weights))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    ce = 0
    frac_correct = 0
    n, m = targets.shape

    ce = (-1/n)*(np.sum((targets*np.log(y)) + ((1-targets)*(np.log(1-y)))))

    for row in range(n):
      #change y to 0's and 1 based on 0.5 threshold
      if (y[row] >= 0.5):
        y[row] = 1
      else:
        y[row] = 0
      #compare target to y
      if (targets[row] == y[row]):
        frac_correct += 1
    frac_correct = frac_correct/n
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    f = None
    df = None

    n = data.shape[0]
    ones = np.ones((n,1))
    Xnew = np.hstack((data,ones))

    #calculate loss
    n = n+1

    f = (-1/n)*(np.sum((targets*np.log(y)) + ((1-targets)*(np.log(1-y)))))
    df = Xnew.T @ (y - targets)/n
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
