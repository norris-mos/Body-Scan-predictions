from pathlib import Path

import numpy as np

def load_data():
    """Load the data from the csv file."""
    data = np.load(Path(__file__).parent.parent / "data/ct_data.npz")
    X_train = data['X_train']; X_val=data['X_val']; X_test=data['X_test']
    y_train = data['y_train']; y_val=data['y_val']; y_test=data['y_test']
    assert X_train.shape[0] == y_train.shape[0],f"dimensions match {X_train.shape[0]}: {y_train.shape[0]}"
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    
    def do_1a():
        print("Train mn: ", y_train.mean()) # verify mean of y_train is zero
        
        N_val = X_val.shape[0]
        std_err_val = X_val.std() / np.sqrt(N_val)
        mn_tr_clipped = X_train[:N_val].mean()
        std_err_tr_clipped = X_train[:N_val].std() / np.sqrt(N_val)

        # report y_val mean with a “standard error”, temporarily assuming that each entry is independent
        print("Val mn: ", y_val.mean())
        print("Val std err: ", std_err_val)

        # report mean with a standard error of first 5,785 entries in y_train
        print("Train mn (clipped data): ", mn_tr_clipped)
        print("Train std err: ", std_err_tr_clipped)

    do_1a()
        
    return X_train, X_val, X_test, y_train, y_val, y_test

def check_means(y_train,y_val):
    y_train_mean = int(np.mean(y_train)); y_train_se = np.std(y_train)
    y_val_mean = np.mean(y_val); y_val_se = np.std(y_val)
    print(f"y_train mean: {y_train_mean} +/- {y_train_se}")
    print(f"y_val mean: {y_val_mean} +/- {y_val_se}")

def error():
    'The error bars are misleading here because each scan comprises of equally distributed sections of the whole body. The error bars imply that most of the data falls within the mean'
    
def preprocess(X_train,X_val,X_test):
    'Preprocess the data'
    
    X_unique = np.unique(X_train, axis=1)
    X_train=X_unique[:, ~np.all(X_unique[1:] == X_unique[:-1], axis=0)]
    X_val_unique = np.unique(X_val, axis=1)
    X_val=X_val_unique[:, ~np.all(X_val_unique[1:] == X_val_unique[:-1], axis=0)]
    X_test_unique  = np.unique(X_test, axis=1)
    X_test=X_test_unique[:, ~np.all(X_test_unique[1:] == X_test_unique[:-1], axis=0)]
    return X_train,X_val,X_test

def find_feat_to_remove(X):
    D = X.shape[1]

    constant = set(np.where(X.std(axis=0) == 0)[0])

    unique_out = np.unique(X, axis=1, return_index=True)
    duplicate = set(np.arange(0, D)) - set(unique_out[1])

    print("Constant: ", constant)
    print("Duplicate: ", duplicate)
    print("Removed: ", sorted(list(constant.union(duplicate))))
    return list(constant.union(duplicate))

def remove_cols(to_remove, X_train, X_test, X_val):
    X_train = np.delete(arr=X_train, obj=to_remove, axis=1)
    X_val = np.delete(arr=X_val, obj=to_remove, axis=1)
    X_test = np.delete(arr=X_test, obj=to_remove, axis=1)
    return X_train, X_val, X_test

def do_1b(X_train, X_val, X_test):
    to_remove = find_feat_to_remove(X=X_train)
    return remove_cols(
        to_remove=to_remove,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test
    )