import gc
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

from lmmvae.pca import LMMPCA
from lmmvae.utils import PCAResult, get_columns_by_prefix, process_one_hot_encoding
from lmmvae.vae import LMMVAE, VAE


def reg_pca_ohe_or_ignore(X_train, X_test, x_cols,
    RE_cols_prefix, d, n_sig2bs, n_sig2bs_spatial, mode, verbose, ignore_RE=False):
    if ignore_RE:
        X_train, X_test = X_train[x_cols], X_test[x_cols]
    else:
        X_train, X_test = process_one_hot_encoding(
            X_train, X_test, x_cols, RE_cols_prefix, mode)
    pca = PCA(n_components=d)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_transformed_tr = pca.fit_transform(X_train)
    X_transformed_te = pca.transform(X_test)
    X_reconstructed_te = pca.inverse_transform(X_transformed_te)
    X_reconstructed_te = scaler.inverse_transform(X_reconstructed_te)

    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    return X_reconstructed_te, [None, none_sigmas, none_sigmas_spatial], None


def reg_lmmpca(X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, verbose, tolerance, max_it, cardinality):
    pca = LMMPCA(n_components=d, max_it=max_it, tolerance=tolerance,
                 cardinality=cardinality, verbose=verbose)
    RE_col = get_columns_by_prefix(X_train, RE_cols_prefix)[0]
    X_transformed_tr = pca.fit_transform(X_train, RE_col=RE_col)
    X_transformed_te = pca.transform(X_test, RE_col=RE_col)
    X_reconstructed_te = None

    sig2bs_mean_est = np.mean(pca.sig2bs_est)
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    return X_reconstructed_te, [pca.sig2e_est, [sig2bs_mean_est], none_sigmas_spatial], pca.n_iter


def reg_vaepca(X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial,
               x_cols, batch_size, epochs, patience, n_neurons, dropout, activation,
               mode, n_sig2bs, beta, verbose, ignore_RE=False):
    if ignore_RE:
        X_train, X_test = X_train[x_cols], X_test[x_cols]
    else:
        X_train, X_test = process_one_hot_encoding(
            X_train, X_test, x_cols, RE_cols_prefix, mode)
    vae = VAE(X_train.shape[1], d, batch_size, epochs, patience, n_neurons,
              dropout, activation, beta, verbose)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    X_transformed_tr = vae.fit_transform(X_train)
    X_transformed_te = vae.transform(X_test)
    X_reconstructed_te = vae.reconstruct(X_transformed_te)

    n_epochs = len(vae.get_history().history['loss'])
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    return X_reconstructed_te, [None, none_sigmas, none_sigmas_spatial], n_epochs


def reg_lmmvae(X_train, X_test, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, x_cols, re_prior, batch_size,
               epochs, patience, n_neurons, dropout, activation, mode, beta, kernel, verbose, U, B_list):
    RE_cols = get_columns_by_prefix(X_train, RE_cols_prefix, mode)
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2', 'longitudinal']:
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2', 't']]
    lmmvae = LMMVAE(mode, X_train[x_cols].shape[1], x_cols, RE_cols, qs, q_spatial,
                    d, n_sig2bs, re_prior, batch_size, epochs, patience, n_neurons,
                    dropout, activation, beta, kernel, verbose)

    # scaler = StandardScaler(with_std=False)
    # X_train_x_cols = pd.DataFrame(scaler.fit_transform(X_train[x_cols]), index=X_train.index, columns=x_cols)
    # X_train = pd.concat([X_train_x_cols, X_train[RE_cols]], axis=1)
    # X_test_x_cols = pd.DataFrame(scaler.transform(X_test[x_cols]), index=X_test.index, columns=x_cols)
    # X_test = pd.concat([X_test_x_cols, X_test[RE_cols]], axis=1)

    X_transformed_tr, B_hat_list, sig2bs_hat_list = lmmvae.fit_transform(X_train, U, B_list)
    X_transformed_te, _, _ = lmmvae.transform(X_test, U, B_list)
    X_reconstructed_te = lmmvae.reconstruct(X_transformed_te, X_test[RE_cols], B_hat_list)

    n_epochs = len(lmmvae.get_history().history['loss'])
    sig2bs_mean_est = [np.mean(sig2bs) for sig2bs in sig2bs_hat_list]
    sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    # TODO: get rid of this
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2']:
        sigmas_spatial = [sig2bs_mean_est[0], None]
        sig2bs_mean_est = []
    return X_reconstructed_te, [None, sig2bs_mean_est, sigmas_spatial], n_epochs


def reg_pca(X_train, X_test, x_cols, RE_cols_prefix, d, pca_type,
            thresh, epochs, qs, q_spatial, n_sig2bs, n_sig2bs_spatial,
            est_cors, batch_size, patience, n_neurons, dropout,
            activation, mode, beta, re_prior, kernel, verbose, U, B_list):
    gc.collect()
    start = time.time()
    if pca_type == 'pca-ignore':
        X_reconstructed_te, sigmas, n_epochs = reg_pca_ohe_or_ignore(
            X_train, X_test, x_cols, RE_cols_prefix, d, n_sig2bs, n_sig2bs_spatial, mode, verbose, ignore_RE=True)
    elif pca_type == 'pca-ohe':
        X_reconstructed_te, sigmas, n_epochs = reg_pca_ohe_or_ignore(
            X_train, X_test, x_cols, RE_cols_prefix, d, n_sig2bs, n_sig2bs_spatial, mode, verbose)
    elif pca_type == 'lmmpca':
        sigmas, n_epochs = reg_lmmpca(
            X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, verbose, thresh, epochs, qs[0])
    elif pca_type == 'vae-ignore':
        X_reconstructed_te, sigmas, n_epochs = reg_vaepca(
            X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, x_cols, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, verbose, ignore_RE=True)
    elif pca_type == 'vae-ohe':
        X_reconstructed_te, sigmas, n_epochs = reg_vaepca(
            X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, x_cols, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, verbose, ignore_RE=False)
    elif pca_type == 'lmmvae':
        X_reconstructed_te, sigmas, n_epochs = reg_lmmvae(
            X_train, X_test, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, x_cols, re_prior, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, beta, kernel, verbose, U, B_list)
    elif pca_type == 'lmmvae-sfc':
        X_reconstructed_te, sigmas, n_epochs = reg_lmmvae(
            X_train, X_test, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, x_cols, re_prior, batch_size,
            epochs, patience, n_neurons, dropout, activation, 'spatial_fit_categorical', beta, kernel, verbose, U, B_list)
    else:
        raise ValueError(f'{pca_type} is an unknown pca_type')
    end = time.time()
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2', 'longitudinal']:
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2', 't']]
    try:
        metric_X = mse(X_test[x_cols].values, X_reconstructed_te[:, :len(x_cols)])
    except:
        metric_X = np.nan
    none_rhos = [None for _ in range(len(est_cors))]
    return PCAResult(metric_X, sigmas, none_rhos, n_epochs, end - start)
