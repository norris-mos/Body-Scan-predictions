import numpy as np

from src.support_code import minimize_list, logreg_cost
from src.q2 import fit_linreg, fit_linreg_gradopt, rmse

def fit_logreg_gradopt(X, yy, alpha):
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(logreg_cost, init, args)
    return ww, bb

def sig(X, ww, b):
    return np.exp(1 / (1 + np.exp(-((X @ ww) + b))))

def do_q3(X_train, y_train, X_val, y_val, X_test, y_test):
    # initialize variables
    N = X_train.shape[0]
    D = X_train.shape[1]
    K = 20
    alp = 30
    prob_train = np.zeros(shape=(X_train.shape[0], K))
    prob_val = np.zeros(shape=(X_val.shape[0], K))
    prob_test = np.zeros(shape=(X_test.shape[0], K))

    # create data store for Q4
    V_q3 = np.zeros(shape=(K, D))
    bk_q3 = np.zeros(shape=(K,))

    K = 20 # number of thresholded classification problems to fit
    mx = np.max(y_train); mn = np.min(y_train); hh = (mx-mn)/(K+1)
    thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)

    for kk in range(K):
        labels = y_train > thresholds[kk]
        ww, bb = fit_logreg_gradopt(X_train, labels, alpha=alp)

        # store predictions for linear regression model training
        prob_train[:, kk] = sig(X=X_train, ww=ww, b=bb)
        prob_val[:, kk] = sig(X=X_val, ww=ww, b=bb)
        prob_test[:, kk] = sig(X=X_test, ww=ww, b=bb)

        # store Q3 results for Q4
        V_q3[kk, :] = ww.copy()
        bk_q3[kk] = bb.copy()

    # train linear regression model: assumed we should include the bias and choose not to regularize it (as before)
    ww, bb = fit_linreg(X=prob_train, yy=y_train, alpha=alp, include_bias=True, regularize_bias=False)
    print("Train RMSE: ", rmse(pred=prob_train @ ww + bb, tar=y_train))
    print("Val RMSE: ", rmse(pred=prob_val @ ww + bb, tar=y_val))
    print("Test RMSE: ", rmse(pred=prob_test @ ww + bb, tar=y_test))

    # store Q3 results for Q4
    ww_q3 = ww.copy()
    bb_q3 = bb.copy()

    return (ww_q3, bb_q3, V_q3, bk_q3)