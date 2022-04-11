import warnings
import numpy as np
from tqdm import tqdm
from fracridge import FracRidgeRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import r2_score, balanced_accuracy_score


def decode_fracridge(X, Y, n_outer_splits=None, cv_outer=None, n_inner_splits=None, cv_inner=None,
                     groups=None):
    """ Predict ("decode") Y from X using fractional ridge regression (FRR) 
    (https://nrdg.github.io/fracridge/). Implements an extended CV scheme
    relative to the FRR implementation, in which a separate frac value *for
    each Y variable* is chosen based on an inner CV loop.
    """

    # Define cross-validate schemes, scaler (/std), and fracridge model
    cv_outer, cv_inner = _define_cv(n_outer_splits, cv_outer, n_inner_splits, cv_inner, groups)
    fracs = np.insert(np.linspace(0.1, 1, num=7, endpoint=True), 0, 0.001)
    frr = FracRidgeRegressor(fracs=fracs, fit_intercept=False)

    if Y.ndim == 1:
        Y = Y[:, None]
    
    if groups is None:
        splits_outer = cv_outer.split(X, Y)
        n_outer_splits = cv_outer.get_n_splits()
    else:
        splits_outer = cv_outer.split(X, Y, groups=groups)
        n_outer_splits = cv_outer.get_n_splits(groups=groups)

    r2_outer = np.zeros((n_outer_splits, Y.shape[1]))

    for i, (train_idx, test_idx) in enumerate(tqdm(splits_outer, total=n_outer_splits)):
        X_train, Y_train = X[train_idx, :], Y[train_idx, :]     
        X_test, Y_test = X[test_idx, :], Y[test_idx, :]

        X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)         
        Y_train = (Y_train - Y_train.mean(axis=0)) / Y_train.std(axis=0)
        Y_test = (Y_test - Y_test.mean(axis=0)) / Y_test.std(axis=0)

        if groups is not None:
            groups_inner = groups[train_idx]
            splits_inner = cv_inner.split(X_train, Y_train, groups=groups_inner)
        else:
            splits_inner = cv_inner.split(X_train, Y_train)

        r2_inner = np.zeros((cv_inner.get_n_splits(), frr.fracs.size, Y.shape[1]))
        for ii, (train_idx_in, test_idx_in) in enumerate(splits_inner):

            X_train_in, Y_train_in = X_train[train_idx_in, :], Y_train[train_idx_in, :]
            X_test_in, Y_test_in = X_train[test_idx_in, :], Y_train[test_idx_in, :]

            X_train_in = (X_train_in - X_train_in.mean(axis=0)) / X_train_in.std(axis=0)
            X_test_in = (X_test_in - X_test_in.mean(axis=0)) / X_test_in.std(axis=0)
            Y_train_in = (Y_train_in - Y_train_in.mean(axis=0)) / Y_train_in.std(axis=0)
            Y_test_in = (Y_test_in - Y_test_in.mean(axis=0)) / Y_test_in.std(axis=0)
            
            # Also cross-validate scaling transform            
            frr.fit(X_train_in, Y_train_in)        
            preds = frr.predict(X_test_in)
            
            num_r2 = np.sum((Y_test_in[:, None, :] - preds) ** 2, axis=0)
            denom_r2 = np.sum((Y_test_in - Y_test_in.mean(axis=0)) ** 2)
            r2_inner[ii, :, :] = 1 - (num_r2 / denom_r2)

        best_frac_idx = np.argmax(np.median(r2_inner, axis=0), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Refit model!
            frr.fit(X_train, Y_train)
            preds = frr.predict(X_test)
    
        # Extract predictions of best frac alpha value per Y value
        preds = np.hstack([preds[:, best_frac_idx[i], i, None] for i in range(Y.shape[1])])
        r2_outer[i, :] = r2_score(Y_test, preds, multioutput='raw_values')
    
    return r2_outer
    

def decode_regular(X, Y, n_outer_splits=None, cv_outer=None, n_inner_splits=None, cv_inner=None,
                   groups=None, classification=False):
    """ Decode feature representations (Y) from a DNN layer (X) using RidgeCV (regression)
    or LogisticRegressionCV (classification). Difference from `fracridge` implementation is that
    here the same alpha parameter is used for each target variable (Y_{i}). """
    
    # -0.001 to 1000
    alphas = np.logspace(-3, 4, num=8, endpoint=True)    

    cv_outer, cv_inner = _define_cv(n_outer_splits, cv_outer, n_inner_splits, cv_inner, groups,
                                    classification=classification)

    if Y.ndim == 1 and not classification:    
        Y = Y[:, None]

    if groups is None:
        splits_outer = cv_outer.split(X, Y)
        n_outer_splits = cv_outer.get_n_splits()
    else:
        splits_outer = cv_outer.split(X, Y, groups=groups)
        n_outer_splits = cv_outer.get_n_splits(groups=groups)

    if classification:
        score_outer = np.zeros(n_outer_splits)
    else:
        score_outer = np.zeros((n_outer_splits, Y.shape[1]))
    
    for i, (train_idx, test_idx) in enumerate(tqdm(splits_outer, total=n_outer_splits)):
        X_train, Y_train = X[train_idx, :], Y[train_idx]
        X_test, Y_test = X[test_idx, :], Y[test_idx]
        
        X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0) 
        
        if not classification:
            Y_train = (Y_train - Y_train.mean(axis=0)) / Y_train.std(axis=0)
            Y_test = (Y_test - Y_test.mean(axis=0)) / Y_test.std(axis=0)
        
        # Note to self: RidgeCV is defined here because it depends on `splits_inner`
        if groups is None:
            splits_inner = cv_inner.split(X_train, Y_train)
        else:
            groups_inner = groups[train_idx]
            splits_inner = cv_inner.split(X_train, Y_train, groups=groups_inner)

        if classification:
            model = LogisticRegressionCV(Cs=1 / alphas, fit_intercept=False,
                                         cv=splits_inner, max_iter=200)             
        else:
            model = RidgeCV(alphas=alphas, fit_intercept=False, cv=splits_inner)

        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        
        if classification:
            score_outer[i] = balanced_accuracy_score(Y_test, preds, adjusted=True)
        else:
            score_outer[i, :] = r2_score(Y_test, preds, multioutput='raw_values')

    return score_outer


def _define_cv(n_outer_splits=None, cv_outer=None, n_inner_splits=None, cv_inner=None,
               groups=None, classification=False):
    
    if n_outer_splits is None:
        n_outer_splits = 4

    if cv_outer is None:
        if groups is None:
            if classification:
                cv_outer = StratifiedKFold(n_splits=n_outer_splits)
            else:
                cv_outer = KFold(n_splits=n_outer_splits)
        else:
            cv_outer = GroupKFold(n_splits=n_outer_splits)
        
    if n_inner_splits is None:
        n_inner_splits = 4

    if cv_inner is None:
        if groups is None:
            if classification:
                cv_inner = StratifiedKFold(n_splits=n_inner_splits)
            else:
                cv_inner = KFold(n_splits=n_inner_splits)
        else:
            cv_inner = GroupKFold(n_splits=n_inner_splits)
    
    return cv_outer, cv_inner    
    
    
if __name__ == '__main__':
    
    N, P = 512, 500
    K = 100
    X = np.random.randn(N, P) + 10
    B = np.random.randn(P, K)
    eps = np.random.randn(N, K)
    Y = X @ B + eps * np.random.randn(K) * 100
    
    #r2 = decode_fracridge(X, Y, cv_standardize=True)
    #print(r2.mean())
    
    r2 = decode_regular(X, Y, cv_standardize=True)
    print(r2.mean())
    