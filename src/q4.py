import numpy as np
import copy
import time

from src.support_code import minimize_list, nn_cost
from src.q2 import rmse


def fit_nn(X, yy, alpha, init):
    """Fits a NN according to nn_cost. Initialisation of parameters must be parsed.

    Args:
        X: Training observations.
        yy: Training labels.
        alpha: Regularization hyperparameter.
        init: Initial state of weights

    Returns:
        Parameters after NN has been fitted.
    """
    args = (X, yy, alpha)
    return minimize_list(nn_cost, init, args)

def glorot_init(n_in, n_out, size, seed = None):
    """Implements Glorot and Bengio’s normalised initialisation as defined in MLP lecture slides. \
        This accounts for the number of inputs and outputs of a given layer.

    Args:
        n_in: Number of inputs.
        n_out: Number of outputs.
        size: Shape of resulting matrix. Parsed to np.random.uniform.
        seed (optional): Sets seed for reproducability. Defaults to None.

    Returns:
        Matrix of random weights. Shape defined by parameter size.
    """
    for_dist = np.sqrt(6 / (n_in + n_out))
    if seed:
        np.random.seed(seed)
    return np.random.uniform(low=-for_dist, high=for_dist, size=size)


def do_q4a(X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    num_iter,  # number of times to train model
    K,  # number of hidden units
    alpha,  # regularization hyperparameter
    seed,  # set seed before random initilisation for reproducibility
    output=True,
):
    N, D = X_train.shape[0], X_train.shape[1]

    # perform random initilisation:
    # use glorot init from MLP lecture slides for weights
    # use 0 for bias params
    ww = glorot_init(n_in=K, n_out=1, size=(K,), seed=seed)  # use seed for reproducibility
    bb = 0
    V = glorot_init(n_in=D, n_out=K, size=(K, D), seed=seed)  # use seed for reproducibility
    bk = np.repeat(0, K)

    random_init = copy.deepcopy((ww, bb, V, bk))

    # train model
    start_time = time.time()
    for i in range(num_iter):  # train multiple times so discussion of training time of methods can be compared
        random_conv = fit_nn(X=X_train, yy=y_train, alpha=alpha, init=random_init)
    if output:
        print(f"Random init trained {num_iter} times: --- %s seconds ---" % (time.time() - start_time))
        print()

    # evaluate model
    E_ran, _ = nn_cost(params=random_conv, X=X_train, yy=y_train, alpha=alpha)
    pred_ran_tr = nn_cost(params=random_conv, X=X_train)
    pred_ran_val = nn_cost(params=random_conv, X=X_val)
    pred_ran_test = nn_cost(params=random_conv, X=X_test)
    if output:
        print("Cost      (random init): ", E_ran)
        print("Tr   RMSE (random init): ", rmse(pred=pred_ran_tr, tar=y_train))
        print("Val  RMSE (random init): ", rmse(pred=pred_ran_val, tar=y_val))
        print("Test RMSE (random init): ", rmse(pred=pred_ran_test, tar=y_test))
        print()

def do_q4b(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    num_iter,  # number of times to train model
    alpha,  # regularization hyperparameter
    params_q3,  # parameters after convergence (Q3)
):
    # set up initilisation of parameters
    warm_start_init = params_q3

    # train model
    start_time = time.time()
    for i in range(num_iter): # train multiple times so discussion of training time of methods can be compared
        warm_start_conv = fit_nn(X=X_train, yy=y_train, alpha=alpha, init=warm_start_init)
    print(f"Warm starts trained {num_iter} times: --- %s seconds ---" % (time.time() - start_time))
    print()

    # evaluate model
    E_warm, _ = nn_cost(params=warm_start_conv, X=X_train, yy=y_train, alpha=alpha)
    pred_warm_tr = nn_cost(params=warm_start_conv, X=X_train)
    pred_warm_val = nn_cost(params=warm_start_conv, X=X_val)
    pred_warm_test = nn_cost(params=warm_start_conv, X=X_test)

    print("Cost      (warm start): ", E_warm)
    print("Tr   RMSE (warm start): ", rmse(pred=pred_warm_tr, tar=y_train))
    print("Val  RMSE (warm start): ", rmse(pred=pred_warm_val, tar=y_val))
    print("Test RMSE (warm start): ", rmse(pred=pred_warm_test, tar=y_test))
    print()

def do_q4(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    params_q3,  # parameters after convergence (Q3)
    num_iter=3,  # number of times to train model
    K=20,  # number of hidden units
    alpha=30,  # reg hyperparam
    seed=64  # set seed before random initilisation for reproducibility
):
    do_q4a(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        alpha=alpha,
        seed=seed,
        num_iter=num_iter,
        K=K
    )
    do_q4b(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        alpha=alpha,
        num_iter=num_iter,
        params_q3=params_q3
    )