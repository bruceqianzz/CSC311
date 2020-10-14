from q2.utils import sigmoid

import numpy as np


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
    N = data.shape[0]
    # add bais col to matrix
    n = np.ones((N,1))
    data = np.column_stack((data,n))
    res = np.dot(data,weights)
    y = 1.0 / (1.0 + np.exp(-res))
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
    left = np.multiply(-1*targets, np.log(y))
    right = np.multiply(1 - targets, np.log(1 - y))
    ce = np.sum(left - right) / y.shape[0]
    classified_correct = np.argwhere(abs(y-targets)<0.5)
    frac_correct = len(classified_correct)/len(y)
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
    N = data.shape[0]
    n = np.ones((N, 1))
    data = np.column_stack((data, n))
    average_loss = evaluate(targets, y)[0]
    f = average_loss
    # df = np.dot(data.transpose(),((-1*targets)/y+(1-targets)/(1-y))*y*(1-y))
    df = np.dot(data.transpose(),y-targets)/N
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

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
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (M+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points (plus a penalty term), gradient of parameters, and the     #
    # probabilities given by penalized logistic regression.             #
    #####################################################################
    N = data.shape[0]
    n = np.ones((N, 1))
    data = np.column_stack((data, n))
    average_loss = evaluate(targets, y)[0]
    penalty = hyperparameters["weight_regularization"] * np.sum(weights*weights)/2
    f = average_loss + penalty
    df = np.dot(data.transpose(), y - targets) / N + np.dot(hyperparameters["weight_regularization"],weights)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


# target = np.ones((10,1))
# target = np.exp(target)
# target2 = np.exp(target)+1
# print(target2*(target/target2).shape[0])
#
