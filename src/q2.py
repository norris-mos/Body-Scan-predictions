import numpy as np

from src.support_code import fit_linreg_gradopt

def rmse(pred, tar):
    return np.sqrt(np.mean((pred - tar)**2))

def get_to_augment(alpha, size, regularize_bias):
    """Produce (D, D) matrix to augment to design matrix for regularization.

    Args:
        alpha: Regularization hyperparameter.
        size: D.
        regularize_bias: Whether to regularize the bias parameter.

    Returns:
        (D, D) matrix.
    """
    # get a vector representing the leading diag
    diag = np.repeat(np.sqrt(alpha), size)
    if not regularize_bias:
        diag[-1] = 0  # final element of diag corresponds to bias regularization

    # create square matrix from vector
    to_augment = np.zeros((size, size), float)
    np.fill_diagonal(to_augment, diag)
    return to_augment

def augment_w_ones(mat):
    """Augments ones to design matrix to facilitate bias parameter during training.
    """
    N = mat.shape[0]
    return np.vstack([mat.T, np.ones(N)]).T
    
def fit_linreg(X, yy, alpha, include_bias=True, regularize_bias=False):
    if include_bias:
        # augment design mat to facilitate bias parameter
        X = augment_w_ones(X)
    N, D = X.shape[0], X.shape[1]

    # augment DxD matrix to force regularization on parameters
    to_augment = get_to_augment(alpha=alpha, size=D, regularize_bias=regularize_bias)
    aug_design_mat = np.vstack([X, to_augment])
    aug_y = np.concatenate([yy, np.zeros(D)])
    
    # run lstsq
    lstsq_out = np.linalg.lstsq(aug_design_mat, aug_y, rcond=None)
    ww = lstsq_out[0]
    return ww[:-1], ww[-1]

def do_q2(X_train, y_train, X_val, y_val):
    alp = 30
    ww, b = fit_linreg(X=X_train, yy=y_train, alpha=alp)
    ww_support, b_support = fit_linreg_gradopt(X=X_train, yy=y_train, alpha=alp)

    y_pred_tr = (X_train @ ww) + b
    y_pred_tr_support = (X_train @ ww_support) + b_support

    y_pred_val = (X_val @ ww) + b
    y_pred_val_support = (X_val @ ww_support) + b_support

    print("Parameters")
    print("fit_linreg: ", ww.mean(), b)
    print("fit_linreg_gradopt: ", ww_support.mean(), b_support)
    print()
    print("RMSE - fit_linreg")
    print("Tr:  ", rmse(pred=y_pred_tr, tar=y_train))
    print("Val: ", rmse(pred=y_pred_val, tar=y_val))
    print()
    print("RMSE - fit_linreg_gradopt")
    print("Tr:  ", rmse(pred=y_pred_tr_support, tar=y_train))
    print("Val: ", rmse(pred=y_pred_val_support, tar=y_val))
    