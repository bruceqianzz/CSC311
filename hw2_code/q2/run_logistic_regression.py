from q2.check_grad import check_grad
from q2.utils import *
from q2.logistic import *
import random

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    print(train_inputs.shape[0])
    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.06,
        "weight_regularization": 0.,
        "num_iterations":100
    }
    weights = np.zeros((M+1,1))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    train = []
    valid = []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights,train_inputs,train_targets,hyperparameters)
        weights -=hyperparameters["learning_rate"]*df
        train.append(f)
        valid.append(evaluate(valid_targets,logistic_predict(weights,valid_inputs))[0])
    plt.plot(range(hyperparameters["num_iterations"]), train,label='training cross entropy')
    plt.plot(range(hyperparameters["num_iterations"]), valid,label="validation cross entropy")
    plt.legend()
    plt.show()
    test_inputs, test_targets = load_test()
    frc = evaluate(test_targets,logistic_predict(weights,test_inputs))[1]
    print(frc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_pen_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Implement the function that automatically evaluates different     #
    # penalty and re-runs penalized logistic regression 5 times.        #
    #####################################################################
    test_inputs, test_targets = load_test()
    hyperparameters = {
        "learning_rate": 0.06,
        "weight_regularization": 0.,
        "num_iterations": 100
    }
    train_ce = []
    train_frac = []
    valid_ce = []
    valid_frac = []
    for j in [0, 0.001, 0.01, 0.1, 1.0]:
        total_train_ce = 0
        total_train_frac = 0
        total_valid_ce = 0
        total_valid_frac = 0
        hyperparameters["weight_regularization"] = j
        r = random.randint(0,4)
        for i in range(5):
            train_plt = []
            valid_plt = []
            weights = np.zeros((M + 1, 1))
            for t in range(hyperparameters["num_iterations"]):
                f, df, y = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                weights -= hyperparameters["learning_rate"] * df
                if(i==r):
                    train_plt.append(f)
                    valid_plt.append(evaluate(valid_targets, logistic_predict(weights, valid_inputs))[0])
            if(i==r):
                plt.plot(range(hyperparameters["num_iterations"]),train_plt,label='train_ce')
                plt.plot(range(hyperparameters["num_iterations"]), valid_plt, label='valid_ce')
                plt.legend()
                plt.show()
                tes_ce, tes_frac = evaluate(test_targets, logistic_predict(weights, test_inputs))
                print("lambd = j : ce = ",tes_ce)
                print("lambd = j : frac = ", tes_frac)
            t_ce, t_frac = evaluate(train_targets,logistic_predict(weights,train_inputs))
            v_ce, v_frac = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
            total_train_ce+=t_ce
            total_train_frac+=t_frac
            total_valid_ce+=v_ce
            total_valid_frac+=v_frac
        train_ce.append(total_train_ce/5)
        train_frac.append(total_train_frac/5)
        valid_ce.append(total_valid_ce/5)
        valid_frac.append(total_valid_frac/5)
    print('train_ce = ',train_ce)
    print('train_frac = ',train_frac)
    print('valid_ce = ',valid_ce)
    print('valid_frac = ',valid_frac)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    # run_logistic_regression()
    run_pen_logistic_regression()
