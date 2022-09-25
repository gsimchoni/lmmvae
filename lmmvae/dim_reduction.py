import gc
import time
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from lmmvae.pca import LMMPCA
from lmmvae.utils import DRResult, get_RE_cols_by_prefix, get_aux_cols, get_q_by_mode, process_one_hot_encoding
from lmmvae.vae import LMMVAE, VAE
from svgpvae.gpvae import SVGPVAE


def run_pca_ohe_or_ignore(X_train, X_test, x_cols,
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


def run_lmmpca(X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, verbose, tolerance, max_it, cardinality):
    pca = LMMPCA(n_components=d, max_it=max_it, tolerance=tolerance,
                 cardinality=cardinality, verbose=verbose)
    RE_col = get_RE_cols_by_prefix(X_train, RE_cols_prefix)[0]
    X_transformed_tr = pca.fit_transform(X_train, RE_col=RE_col)
    X_transformed_te = pca.transform(X_test, RE_col=RE_col)
    X_reconstructed_te = None

    sig2bs_mean_est = np.mean(pca.sig2bs_est)
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    return X_reconstructed_te, [pca.sig2e_est, [sig2bs_mean_est], none_sigmas_spatial], pca.n_iter


def run_vae(X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial,
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


def run_lmmvae(X_train, X_test, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, x_cols, re_prior, batch_size,
               epochs, patience, n_neurons, n_neurons_re, dropout, activation, mode, beta, kernel, verbose, U, B_list):
    RE_cols = get_RE_cols_by_prefix(X_train, RE_cols_prefix, mode)
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2', 'longitudinal']:
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2', 't']]
    lmmvae = LMMVAE(mode, X_train[x_cols].shape[1], x_cols, RE_cols, qs, q_spatial,
                    d, n_sig2bs, re_prior, batch_size, epochs, patience, n_neurons, n_neurons_re,
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


def run_svgpvae(X_train, X_test, x_cols, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, mode,
    batch_size, epochs, patience, n_neurons, dropout, activation, beta, M, nr_inducing_points, nr_inducing_per_unit, verbose, scale=False):
    RE_cols = get_RE_cols_by_prefix(X_train, RE_cols_prefix, mode, pca_type='svgpvae')
    aux_cols = get_aux_cols(mode)
    q = get_q_by_mode(qs, q_spatial, mode)
    svgpvae = SVGPVAE(d, q, x_cols, batch_size, epochs, patience, n_neurons, dropout, activation, verbose,
        M, nr_inducing_points, nr_inducing_per_unit, RE_cols, aux_cols, beta, GECO=False, disable_gpu=True)

    if scale:
        x_cols_pca = [col for col in x_cols if col not in RE_cols + aux_cols]
        scaler = MinMaxScaler()
        X_train_x_cols = pd.DataFrame(scaler.fit_transform(X_train[x_cols_pca]), index=X_train.index, columns=x_cols_pca)
        X_train = pd.concat([X_train_x_cols, X_train[RE_cols + aux_cols]], axis=1)
        X_test_x_cols = pd.DataFrame(scaler.transform(X_test[x_cols_pca]), index=X_test.index, columns=x_cols_pca)
        X_test = pd.concat([X_test_x_cols, X_test[RE_cols + aux_cols]], axis=1)

    X_transformed_tr, X_transformed_te, X_reconstructed_te = svgpvae.run(X_train, X_test)

    if scale:
        X_reconstructed_te = scaler.inverse_transform(X_reconstructed_te)
    
    n_epochs = svgpvae.get_n_epochs()
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    return X_reconstructed_te, [None, none_sigmas, none_sigmas_spatial], n_epochs


def run_gppvae(X_train, X_test, x_cols, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, mode,
    batch_size, epochs, patience, n_neurons, dropout, activation, verbose, scale=True):
#     RE_cols = get_RE_cols_by_prefix(X_train, RE_cols_prefix, mode, pca_type='gppvae')
#     if mode == 'categorical':
#         aux_cols = []
#         q = qs[0]
#     elif mode in ['spatial', 'spatial_fit_categorical', 'spatial2']:
#         aux_cols = ['D1', 'D2']
#         q = q_spatial
#     elif mode == 'longitudinal':
#         aux_cols = ['t']
#         q = qs[0]
#     else:
#         raise ValueError(f'mode {mode} not recognized')
    
#     if scale:
#         x_cols_pca = [col for col in x_cols if col not in RE_cols + aux_cols]
#         scaler = MinMaxScaler()
#         X_train_x_cols = pd.DataFrame(scaler.fit_transform(X_train[x_cols_pca]), index=X_train.index, columns=x_cols_pca)
#         X_train = pd.concat([X_train_x_cols, X_train[RE_cols + aux_cols]], axis=1)
#         X_test_x_cols = pd.DataFrame(scaler.transform(X_test[x_cols_pca]), index=X_test.index, columns=x_cols_pca)
#         X_test = pd.concat([X_test_x_cols, X_test[RE_cols + aux_cols]], axis=1)
    
#     # split train to train and eval
#     X_train_new, X_eval_new = train_test_split(X_train, test_size=0.1)

#     # get dictionaries
#     M = 10
#     train_data_dict, eval_data_dict, test_data_dict = process_data_for_svgpvae(
#         X_train_new, X_test, X_eval_new, x_cols, aux_cols, RE_cols, M, add_train_index_aux=True, sort_train=True)
#     train_ids_mask = get_train_ids_mask(train_data_dict, aux_cols)
    
#     # run GPPVAE
#     X_reconstructed_te, n_epochs = run_experiment_GPPVAE(train_data_dict, eval_data_dict, test_data_dict, train_ids_mask,
#         d, q, batch_size, epochs, patience, n_neurons, dropout, activation, verbose, elbo_arg='GPVAE_Casale',
#         M = M, RE_cols=RE_cols, aux_cols=aux_cols)
    
#     if scale:
#         X_reconstructed_te = scaler.inverse_transform(X_reconstructed_te)
    
#     none_sigmas = [None for _ in range(n_sig2bs)]
#     none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
#     return X_reconstructed_te, [None, none_sigmas, none_sigmas_spatial], n_epochs
    pass


def reg_dr(X_train, X_test, x_cols, RE_cols_prefix, d, dr_type,
            thresh, epochs, qs, q_spatial, n_sig2bs, n_sig2bs_spatial,
            est_cors, batch_size, patience, n_neurons, n_neurons_re, dropout,
            activation, mode, beta, re_prior, kernel, verbose, U, B_list):
    gc.collect()
    start = time.time()
    if dr_type == 'pca-ignore':
        X_reconstructed_te, sigmas, n_epochs = run_pca_ohe_or_ignore(
            X_train, X_test, x_cols, RE_cols_prefix, d, n_sig2bs, n_sig2bs_spatial, mode, verbose, ignore_RE=True)
    elif dr_type == 'pca-ohe':
        X_reconstructed_te, sigmas, n_epochs = run_pca_ohe_or_ignore(
            X_train, X_test, x_cols, RE_cols_prefix, d, n_sig2bs, n_sig2bs_spatial, mode, verbose)
    elif dr_type == 'lmmpca':
        sigmas, n_epochs = run_lmmpca(
            X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, verbose, thresh, epochs, qs[0])
    elif dr_type == 'vae-ignore':
        X_reconstructed_te, sigmas, n_epochs = run_vae(
            X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, x_cols, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, verbose, ignore_RE=True)
    elif dr_type == 'vae-ohe':
        X_reconstructed_te, sigmas, n_epochs = run_vae(
            X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, x_cols, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, verbose, ignore_RE=False)
    elif dr_type == 'lmmvae':
        X_reconstructed_te, sigmas, n_epochs = run_lmmvae(
            X_train, X_test, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, x_cols, re_prior, batch_size,
            epochs, patience, n_neurons, n_neurons_re, dropout, activation, mode, beta, kernel, verbose, U, B_list)
    elif dr_type == 'lmmvae-sfc':
        X_reconstructed_te, sigmas, n_epochs = run_lmmvae(
            X_train, X_test, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, x_cols, re_prior, batch_size,
            epochs, patience, n_neurons, n_neurons_re, dropout, activation, 'spatial_fit_categorical', beta, kernel, verbose, U, B_list)
    elif dr_type.startswith('svgpvae'):
        dr_type_split = dr_type.split('-')
        nr_inducing_points, nr_inducing_per_unit = int(dr_type_split[1]), int(dr_type_split[2])
        X_reconstructed_te, sigmas, n_epochs = run_svgpvae(
            X_train, X_test, x_cols, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, mode, batch_size,
            epochs, patience, n_neurons, dropout, activation, beta, M=10, nr_inducing_points=nr_inducing_points,
            nr_inducing_per_unit=nr_inducing_per_unit, verbose=verbose)
    elif dr_type == 'gppvae':
        X_reconstructed_te, sigmas, n_epochs = run_gppvae(
            X_train, X_test, x_cols, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, mode, batch_size,
            epochs, patience, n_neurons, dropout, activation, verbose)
    else:
        raise ValueError(f'{dr_type} is an unknown dr_type')
    end = time.time()
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2', 'longitudinal']:
        x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2', 't']]
    try:
        metric_X = mse(X_test[x_cols].values, X_reconstructed_te[:, :len(x_cols)])
    except:
        metric_X = np.nan
    none_rhos = [None for _ in range(len(est_cors))]
    return DRResult(metric_X, sigmas, none_rhos, n_epochs, end - start)
