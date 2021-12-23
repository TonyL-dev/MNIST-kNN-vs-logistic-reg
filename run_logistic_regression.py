from q2.check_grad import check_grad
from q2.utils import *
from q2.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.01,
        "weight_regularization": 0.,
        "num_iterations": 500
    }
    weights = np.zeros((M+1, 1))
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
    cost = []
    cost_val = []
    iteration = []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, None)
        weights = weights - (hyperparameters["learning_rate"]*df)
        if (t % 100 == 0):
          cost.append(f)
          iteration.append(t)

          y = logistic_predict(weights, valid_inputs)
          ce, frac_correct = evaluate(valid_targets, y)
          cost_val.append(ce)
    
    plt.plot(iteration, cost, label = "Training")
    plt.plot(iteration, cost_val, label = "validation")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost")
    plt.legend()
    plt.title("Cost vs iteration number")
    plt.show()
    ### Uncomment as necessary!
    #for training
    #evaluate(train_targets, y)

    #for validation
    #y = logistic_predict(weights, valid_inputs)
    #evaluate(valid_targets, y)

    #for test
    #y = logistic_predict(weights,test_inputs)
    #evaluate(test_targets, y)
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
    run_logistic_regression(
