import copy

import numpy as np
from scipy import stats

from src.q4 import glorot_init, fit_nn
from src.q3 import rmse
from src.support_code import nn_cost, gp_post_par

def prob_of_improv_acq_func(max_ob, mean, sd):
    """Prob of improvement calculation for N observations. 

    Args:
        max_ob : _description_
        mean (N,): GP posterior mean at each training location 
        sd (N,): GP posterior standard deviation (not variance!) at each training location

    Returns:
        Probability of improvement matrix
    """
    density_input = (mean - max_ob) / sd
    return stats.norm.cdf(density_input) 

def train_nn_reg(alpha, X_train, y_train, X_val, y_val, num_iter=1):
    """Given a regularization hyperparameter alpha, train neural network from Q4 and return RMSE on validation set.

    Args:
        alpha: Regularization hyperparameter
        X_train
        y_train
        X_val
        y_val
        num_iter (optional): Number of times to produce RMSE for given model configuration. Defaults to 1.

    Returns:
        Average RMSE of model with given configuration.
    """
    N = X_train.shape[0]
    D = X_train.shape[1]
    K = 20
    
    def train_single_model():
        # perform random initialisation
        ww = glorot_init(n_in=K, n_out=1, size=(K,))
        bb = 0
        V = glorot_init(n_in=D, n_out=K, size=(K, D))
        bk = np.repeat(0, 20)
        random_init = (ww, bb, V, bk)

        # train model and perform prediction
        conv_params = fit_nn(X=X_train, yy=y_train, alpha=alpha, init=random_init)
        return rmse(pred=nn_cost(params=conv_params, X=X_val), tar=y_val)

    # train multiple times to get an average RMSE
    return np.mean([train_single_model() for i in range(num_iter)])

def init_gp(
    base,  # baseline for comparison: RMSE produced in Q4a
    X_train,
    y_train,
    X_val,
    y_val,
    num_iter,  # number of times to generate RMSE (for average)
    K_train_loc = np.asarray([13, 26, 39]),  # choose 3 training locations
):
    """Initialise the GP. Pick 3 training locations and get RMSE at each.

    Returns:
        train_loc: Locations of GP training points.
        test_loc: Locations of GP test points.
        train_obs: Observed RMSE values at training locations.
    """
    test_loc = np.arange(0, 50, 0.02)  # create all possible values of alpha
    train_obs = np.zeros(shape=train_loc.shape)

    for i, loc in enumerate(train_loc):
        # train model for given alpha (defined by loc)
        rmse_at_loc = train_nn_reg(
            alpha=loc,
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

def do_q5(X_train, y_train, X_val, y_val, num_iter=3):
    base = np.log(0.27073195313011955)

    # set up locations and train at 3 points
    print("Setting up GP")
    train_loc, test_loc, train_obs = init_gp(
        base=base,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_loc = np.asarray([7, 26, 41]),  # choose 3 training locations
        num_iter=num_iter,
    )

    for i in range(5):
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

    for i, (loc, err) in enumerate(zip(train_loc, train_obs_rmse)):
        print(i, loc, err)

    print()
    print("Best model")
    print(train_loc[ind], train_obs_rmse[ind])