import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

from lmmpca.pca import LMMPCA
from lmmpca.utils import PCAResult, process_one_hot_encoding
from lmmpca.vaepca import LMMVAE, VAE


def reg_pca_ohe_or_ignore(X_train, X_test, y_train, y_test, x_cols, RE_col, d, verbose, ignore_RE=False):
    if ignore_RE:
        X_train, X_test = X_train[x_cols], X_test[x_cols]
    else:
        X_train, X_test = process_one_hot_encoding(
            X_train, X_test, x_cols, RE_col)
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


def reg_vaepca(X_train, X_test, y_train, y_test, RE_col, d,
               x_cols, batch_size, epochs, patience, n_neurons, dropout, activation,
               verbose, ignore_RE=False):
    if ignore_RE:
        X_train, X_test = X_train[x_cols], X_test[x_cols]
    else:
        X_train, X_test = process_one_hot_encoding(
            X_train, X_test, x_cols, RE_col)
    vae = VAE(X_train.shape[1], d, batch_size, epochs, patience, n_neurons,
              dropout, activation, verbose)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    X_transformed_tr = vae.fit_transform(X_train)
    X_transformed_te = vae.transform(X_test)

    lm_fit = LinearRegression().fit(X_transformed_tr, y_train)
    y_pred = lm_fit.predict(X_transformed_te)
    n_epochs = len(vae.get_history().history['loss'])
    return y_pred, [None, None], n_epochs


def reg_lmmvae(X_train, X_test, y_train, y_test, RE_col, q, d, x_cols, re_prior, batch_size,
               epochs, patience, n_neurons, dropout, activation, verbose, U, B):
    lmmvae = LMMVAE(X_train[x_cols].shape[1], x_cols, RE_col, q, d, re_prior, batch_size, epochs, patience, n_neurons,
                    dropout, activation, verbose)

    X_transformed_tr = lmmvae.fit_transform(X_train, U, B)
    X_transformed_te = lmmvae.transform(X_test, U, B)

    lm_fit = LinearRegression().fit(X_transformed_tr, y_train)
    y_pred = lm_fit.predict(X_transformed_te)
    n_epochs = len(lmmvae.get_history().history['loss'])
    return y_pred, [None, None], n_epochs


def reg_pca(X_train, X_test, y_train, y_test, x_cols, RE_col, d, pca_type,
            thresh, epochs, cardinality, batch_size, patience, n_neurons, dropout,
            activation, verbose, U, B):
    start = time.time()
    if pca_type == 'ignore':
        y_pred, sigmas, n_epochs = reg_pca_ohe_or_ignore(
            X_train, X_test, y_train, y_test, x_cols, RE_col, d, verbose, ignore_RE=True)
    elif pca_type == 'ohe':
        y_pred, sigmas, n_epochs = reg_pca_ohe_or_ignore(
            X_train, X_test, y_train, y_test, x_cols, RE_col, d, verbose)
    elif pca_type == 'lmmpca':
        y_pred, sigmas, n_epochs = reg_lmmpca(
            X_train, X_test, y_train, y_test, RE_col, d, verbose, thresh, epochs, cardinality)
    elif pca_type == 'vae':
        y_pred, sigmas, n_epochs = reg_vaepca(
            X_train, X_test, y_train, y_test, RE_col, d, x_cols, batch_size,
            epochs, patience, n_neurons, dropout, activation, verbose, ignore_RE=False)
    elif pca_type == 'lmmvae':
        y_pred, sigmas, n_epochs = reg_lmmvae(
            X_train, X_test, y_train, y_test, RE_col, cardinality, d, x_cols, 1.0, batch_size,
            epochs, patience, n_neurons, dropout, activation, verbose, U, B)
    else:
        raise ValueError(f'{pca_type} is an unknown pca_type')
    end = time.time()
    metric = mse(y_test, y_pred)
    return PCAResult(metric, sigmas, n_epochs, end - start)
