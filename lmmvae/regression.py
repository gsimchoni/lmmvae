import gc
import time
import torch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lmmvae.gplvm import GPLVM

from lmmvae.pca import LMMPCA
from lmmvae.utils import DRResult, get_columns_by_prefix, process_one_hot_encoding
from lmmvae.vae import LMMVAE, VAE


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
    RE_col = get_columns_by_prefix(X_train, RE_cols_prefix)[0]
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
    RE_cols = get_columns_by_prefix(X_train, RE_cols_prefix, mode)
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


def run_gplvm(X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, x_cols, batch_size,
    epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, verbose):

    X_train = X_train[x_cols]
    X_test = X_test[x_cols]

    # x_cols_mlp = [col for col in x_cols if col not in ['D1', 'D2']]
    # x_cols_gp = ['D1', 'D2']

    # X_train, X_valid = train_test_split(X_train, test_size=0.1)
    
    # not needed later
    X_train = torch.Tensor(X_train.values)
    # valid_x = torch.Tensor(X_valid.values)
    X_test = torch.Tensor(X_test.values)
    
    # train_x_mlp = torch.Tensor(X_train[x_cols_mlp].values)
    # train_x_gp = torch.Tensor(X_train[x_cols_gp].values)
    # valid_x_mlp = torch.Tensor(X_valid[x_cols_mlp].values)
    # valid_x_gp = torch.Tensor(X_valid[x_cols_gp].values)
    # test_x_mlp = torch.Tensor(X_test[x_cols_mlp].values)
    # test_x_gp = torch.Tensor(X_test[x_cols_gp].values)

    # not needed later
    if torch.cuda.is_available():
            X_train, X_test = X_train.cuda(), X_test.cuda()

    # if torch.cuda.is_available():
    #     train_x_mlp, train_x_gp, valid_x_mlp, valid_x_gp, test_x_mlp, test_x_gp = train_x_mlp.cuda(), train_x_gp.cuda(), valid_x_mlp.cuda(), valid_x_gp.cuda(), test_x_mlp.cuda(), test_x_gp.cuda()

    # train_dataset = torch.utils.data.TensorDataset(train_x_mlp, train_x_gp)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # valid_dataset = torch.utils.data.TensorDataset(valid_x_mlp, valid_x_gp)
    # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # test_dataset = torch.utils.data.TensorDataset(test_x_mlp, test_x_gp)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_inducing = 500
    gplvm = GPLVM(X_train.shape[0], X_train.shape[1], d, n_inducing, batch_size, epochs, patience, n_neurons,
        dropout, activation, verbose)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    X_transformed_tr = gplvm.fit_transform(X_train)
    X_transformed_te = gplvm.transform(X_test)
    X_reconstructed_te = gplvm.reconstruct(X_transformed_te)

    n_epochs = len(gplvm.get_history())
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    return X_reconstructed_te, [None, none_sigmas, none_sigmas_spatial], n_epochs


def run_dim_reduction(X_train, X_test, x_cols, RE_cols_prefix, d, dr_type,
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
    elif dr_type == 'gplvm':
        X_reconstructed_te, sigmas, n_epochs = run_gplvm(
            X_train, X_test, RE_cols_prefix, d, n_sig2bs_spatial, x_cols, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, verbose)
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
