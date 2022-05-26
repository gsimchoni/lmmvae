import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

from lmmpca.pca import LMMPCA
from lmmpca.utils import PCAResult, process_one_hot_encoding


def reg_pca_ohe_or_ignore(X_train, X_test, y_train, y_test, x_cols, RE_col, d, verbose, ignore_RE=False):
    if ignore_RE:
        X_train, X_test = X_train[x_cols], X_test[x_cols]
    else:
        X_train, X_test = process_one_hot_encoding(X_train, X_test, x_cols, RE_col)
    pca = PCA(n_components=d)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_transformed_tr = pca.fit_transform(X_train)
    X_transformed_te = pca.transform(X_test)

    lm_fit = LinearRegression().fit(X_transformed_tr, y_train)
    y_pred = lm_fit.predict(X_transformed_te)
    return y_pred, [None, None], None


def reg_lmmpca(X_train, X_test, y_train, y_test, RE_col, d, verbose, tolerance, max_it, cardinality):
    pca = LMMPCA(n_components=d, max_it=max_it, tolerance=tolerance,
            cardinality=cardinality, verbose=verbose)

    X_transformed_tr = pca.fit_transform(X_train, RE_col=RE_col)
    X_transformed_te = pca.transform(X_test, RE_col=RE_col)

    lm_fit = LinearRegression().fit(X_transformed_tr, y_train)
    y_pred = lm_fit.predict(X_transformed_te)
    sig2bs_mean_est = np.mean(pca.sig2bs_est)
    return y_pred, [pca.sig2e_est, sig2bs_mean_est], pca.n_iter


def reg_pca(X_train, X_test, y_train, y_test, x_cols, RE_col, d, pca_type, thresh, max_it, cardinality, verbose):
    start = time.time()
    if pca_type == 'ignore':
        y_pred, sigmas, n_epochs = reg_pca_ohe_or_ignore(
            X_train, X_test, y_train, y_test, x_cols, RE_col, d, verbose, ignore_RE=True)
    elif pca_type == 'ohe':
        y_pred, sigmas, n_epochs = reg_pca_ohe_or_ignore(
            X_train, X_test, y_train, y_test, x_cols, RE_col, d, verbose)
    elif pca_type == 'lmmpca':
        y_pred, sigmas, n_epochs = reg_lmmpca(
            X_train, X_test, y_train, y_test, RE_col, d, verbose, thresh, max_it, cardinality)
    else:
        raise ValueError(f'{pca_type} is an unknown pca_type')
    end = time.time()
    metric = mse(y_test, y_pred)
    return PCAResult(metric, sigmas, n_epochs, end - start)
