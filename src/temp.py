import copy
import numpy as np
from scipy.optimize import minimize

from src.support_code import params_unwrap, params_wrap, nn_cost, gp_post_par
from src.q2 import rmse
from src.q4 import fit_nn
from src.q5 import glorot_init, train_nn_reg, prob_of_improv_acq_func


def minimize_list_custom_iter(cost, init_list, args, iter):
    """Optimize a list of arrays (wrapper of scipy.optimize.minimize)

    The input function "cost" should take a list of parameters,
    followed by any extra arguments:
        cost(init_list, *args)
    should return the cost of the initial condition, and a list in the same
    format as init_list giving gradients of the cost wrt the parameters.

    The options to the optimizer have been hard-coded. You may wish
    to change disp to True to get more diagnostics. You may want to
    decrease maxiter while debugging. Although please report all results
    in Q2-5 using maxiter=500.
    """
    opt = {'maxiter': iter, 'disp': False}
    init, unwrap = params_wrap(init_list)
    def wrap_cost(vec, *args):
        E, params_bar = cost(unwrap(vec), *args)
        vec_bar, _ = params_wrap(params_bar)
        return E, vec_bar
    res = minimize(wrap_cost, init, args, 'L-BFGS-B', jac=True, options=opt)
    return unwrap(res.x)

def fit_nn_custom_iter(X, yy, alpha, init, max_iter):
    """Fits a NN according to nn_cost, for a fixed number of iterations. \
        Initialisation of parameters must be parsed.

    Args:
        X: Training observations.
        yy: Training labels.
        alpha: Regularization hyperparameter.
        init: Initial state of weights
        max_iter: Max number of iterations for optimizer.

    Returns:
        Parameters after NN has been fitted.
    """
    args = (X, yy, alpha)
    return minimize_list_custom_iter(nn_cost, init, args, iter=max_iter)


def do_q6(X_train, y_train, X_val, y_val):
    alpha = 3.98  # optimal alpha found in Q5
    K = 20
    N, D = X_train.shape[0], X_train.shape[1]

    # perform init
    ww = glorot_init(n_in=K, n_out=1, size=(K,))
    bb = 0
    V = glorot_init(n_in=D, n_out=K, size=(K, D))
    bk = np.repeat(0, 20)
    params = copy.deepcopy((ww, bb, V, bk))
    
    first = True
    rmse___, rmse__, rmse_ = 10000000, 10000000 - 1, 10000000 - 2
    i = 0

    min_rmse_ = 1
    #while ((i < 3) or ((rmse__ > rmse_) or (rmse___ > rmse_))):
    while ((i < 3) or (rmse_ < (min_rmse_ * 1.05))):
        #rmse___ = copy.copy(rmse__)
        #rmse__ = copy.copy(rmse_)

        params = fit_nn_custom_iter(X=X_train, yy=y_train, alpha=alpha, init=params, max_iter=50)
        rmse_ = rmse(pred=nn_cost(params, X=X_val), tar=y_val)
        #print(i, "Val RMSE", rmse_, rmse__, rmse___)
        print(i, "Val RMSE", rmse_, min_rmse_)

        if rmse_ < min_rmse_:
            min_rmse_ = rmse_
        i+= 1

    return params



def do_q6b(X_train, y_train, X_val, y_val):
    alpha = 3.98  # optimal alpha found in Q5
    K = 128
    N, D = X_train.shape[0], X_train.shape[1]

    # perform init
    ww = glorot_init(n_in=K, n_out=1, size=(K,))
    bb = 0
    V = glorot_init(n_in=D, n_out=K, size=(K, D))
    bk = np.repeat(0, 20)
    params = copy.deepcopy((ww, bb, V, bk))
    
    first = True
    rmse___, rmse__, rmse_ = 10000000, 10000000 - 1, 10000000 - 2
    i = 0

    min_rmse_ = 1
    #while ((i < 3) or ((rmse__ > rmse_) or (rmse___ > rmse_))):
    while ((i < 3) or (rmse_ < (min_rmse_ * 1.05))):
        #rmse___ = copy.copy(rmse__)
        #rmse__ = copy.copy(rmse_)

        params = fit_nn_custom_iter(X=X_train, yy=y_train, alpha=alpha, init=params, max_iter=50)
        rmse_ = rmse(pred=nn_cost(params, X=X_val), tar=y_val)
        #print(i, "Val RMSE", rmse_, rmse__, rmse___)
        print(i, "Val RMSE", rmse_, min_rmse_)

        if rmse_ < min_rmse_:
            min_rmse_ = rmse_
        i+= 1

    return params


        

def train_nn_reg_q6(K, X_train, y_train, X_val, y_val, num_iter=1):
    """Given a regularization hyperparameter alpha, train neural network from Q4 and return RMSE on validation set.

    Args:
        K: Number of hidden units
        X_train
        y_train
        X_val
        y_val
        num_iter (optional): Number of times to produce RMSE for given model configuration. Defaults to 1.

    Returns:
        Average RMSE of model with given configuration.
    """
    D = X_train.shape[1]
    
    def train_single_model():
        # perform random initialisation
        ww = glorot_init(n_in=K, n_out=1, size=(K,))
        bb = 0
        V = glorot_init(n_in=D, n_out=K, size=(K, D))
        bk = np.repeat(0, K)
        random_init = (ww, bb, V, bk)

        # train model and perform prediction
        conv_params = fit_nn(X=X_train, yy=y_train, alpha=0, init=random_init)
        return rmse(pred=nn_cost(params=conv_params, X=X_val), tar=y_val)

    # train multiple times to get an average RMSE
    return np.mean([train_single_model() for i in range(num_iter)])



def init_gp_k_alpha(
    base,  # baseline for comparison: RMSE produced in Q4a
    X_train,
    y_train,
    X_val,
    y_val,
    num_iter,  # number of times to generate RMSE (for average)
    k_train_loc = np.log([20, 32, 32, 128, 128]), 
    alpha_train_loc=np.asarray([3.98, 7, 7, 2, 2]) 
):
    """Initialise the GP. Pick 3 training locations and get RMSE at each.

    Returns:
        train_loc: Locations of GP training points.
        test_loc: Locations of GP test points.
        train_obs: Observed RMSE values at training locations.
    """
    train_loc = np.vstack([k_train_loc, alpha_train_loc]).T

    K_test = np.asarray([10, 16, 20, 32, 64, 128])
    K_test = K_test.reshape(len(K_test), 1)
    alpha_test = np.arange(0, 10, 0.2)
    alpha_test = alpha_test.reshape(len(alpha_test), 1)

    train_obs = np.zeros(shape=(train_loc.shape)

    for i, loc in enumerate(train_loc):
        # train model for given alpha (defined by loc)
        rmse_at_loc = train_nn_reg_q6(
            K=int(np.exp(loc)),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_iter=num_iter,
        )
        
        # store error
        train_obs[i] = base - np.log(rmse_at_loc)

        # remove chosen alpha from test locations
        test_loc = np.delete(arr=test_loc, obj=np.where(test_loc==loc))
    return train_loc, test_loc, train_obs


def do_q6_k_alp_baye_opt(X_train, y_train, X_val, y_val, base, train_loc, test_loc, train_obs, num_iter=3):
    # set up locations and train at 3 points
    for i in range(3):
        print("Iteration ", i)
        # get parameters of GP
        mu, sig = gp_post_par(
            X_obs=train_loc,
            X_rest=test_loc,
            yy=train_obs,
            # use defaults for sigma_y, ell, sigma_f
        )
        
        # find which location offers best improvement
        improv = prob_of_improv_acq_func(max_ob=train_obs.max(), mean=mu, sd=np.sqrt(sig.diagonal()))
        ind = improv.argmax()
        new_train_loc = test_loc[ind]
        print("New location and prob of improvement ", test_loc[ind], improv.max())

        # train at chosen location
        rmse_at_loc = train_nn_reg(
            alpha=new_train_loc,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_iter=num_iter,
        )
        print("Trained model")
        print()
        # add most recent point to train points and remove from test points
        train_obs = np.append(train_obs, base - np.log(rmse_at_loc))
        train_loc = np.append(train_loc, new_train_loc)
        test_loc = np.delete(arr=test_loc, obj=ind)

    # output summary
    train_obs_rmse = np.exp(base - train_obs)
    ind = train_obs_rmse.argmin()


    print()
    print("Best model")
    print(train_loc[ind], train_obs_rmse[ind])