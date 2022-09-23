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
from lmmvae.utils import DRResult, get_columns_by_prefix, process_one_hot_encoding
from lmmvae.vae import LMMVAE, VAE

# TODO: Apparently these imports fail other VAE methods, probably because of TF1
from SVGPVAE.TABULAR_experiment import run_experiment_SVGPVAE, run_experiment_GPPVAE


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


def run_svgpvae(X_train, X_test, x_cols, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, mode,
    batch_size, epochs, patience, n_neurons, dropout, activation, verbose, scale=False):
    RE_cols = get_columns_by_prefix(X_train, RE_cols_prefix, mode, pca_type='svgpvae')
    if mode == 'categorical':
        aux_cols = []
        q = qs[0]
    elif mode in ['spatial', 'spatial_fit_categorical', 'spatial2']:
        aux_cols = ['D1', 'D2']
        q = q_spatial
    elif mode == 'longitudinal':
        aux_cols = ['t']
        q = qs[0]
    else:
        raise ValueError(f'mode {mode} not recognized')
    
    if scale:
        x_cols_pca = [col for col in x_cols if col not in RE_cols + aux_cols]
        scaler = MinMaxScaler()
        X_train_x_cols = pd.DataFrame(scaler.fit_transform(X_train[x_cols_pca]), index=X_train.index, columns=x_cols_pca)
        X_train = pd.concat([X_train_x_cols, X_train[RE_cols + aux_cols]], axis=1)
        X_test_x_cols = pd.DataFrame(scaler.transform(X_test[x_cols_pca]), index=X_test.index, columns=x_cols_pca)
        X_test = pd.concat([X_test_x_cols, X_test[RE_cols + aux_cols]], axis=1)
    
    # split train to train and eval
    X_train_new, X_eval_new = train_test_split(X_train, test_size=0.1)

    # get dictionaries
    M = 10
    nr_inducing_points = 16
    nr_inducing_per_unit = 2
    train_data_dict, eval_data_dict, test_data_dict = process_data_for_svgpvae(
        X_train_new, X_test, X_eval_new, x_cols, aux_cols, RE_cols, M)
    
    # run SVGPVAE
    X_reconstructed_te, n_epochs = run_experiment_SVGPVAE(train_data_dict, eval_data_dict, test_data_dict,
        d, q, batch_size, epochs, patience, n_neurons, dropout, activation, verbose, elbo_arg='SVGPVAE_Hensman',
        M = M, nr_inducing_units=nr_inducing_points, nr_inducing_per_unit = nr_inducing_per_unit,
        RE_cols=RE_cols, aux_cols=aux_cols, GECO=False)
    
    if scale:
        X_reconstructed_te = scaler.inverse_transform(X_reconstructed_te)
    
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    return X_reconstructed_te, [None, none_sigmas, none_sigmas_spatial], n_epochs


def run_gppvae(X_train, X_test, x_cols, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, mode,
    batch_size, epochs, patience, n_neurons, dropout, activation, verbose, scale=True):
    RE_cols = get_columns_by_prefix(X_train, RE_cols_prefix, mode, pca_type='gppvae')
    if mode == 'categorical':
        aux_cols = []
        q = qs[0]
    elif mode in ['spatial', 'spatial_fit_categorical', 'spatial2']:
        aux_cols = ['D1', 'D2']
        q = q_spatial
    elif mode == 'longitudinal':
        aux_cols = ['t']
        q = qs[0]
    else:
        raise ValueError(f'mode {mode} not recognized')
    
    if scale:
        x_cols_pca = [col for col in x_cols if col not in RE_cols + aux_cols]
        scaler = MinMaxScaler()
        X_train_x_cols = pd.DataFrame(scaler.fit_transform(X_train[x_cols_pca]), index=X_train.index, columns=x_cols_pca)
        X_train = pd.concat([X_train_x_cols, X_train[RE_cols + aux_cols]], axis=1)
        X_test_x_cols = pd.DataFrame(scaler.transform(X_test[x_cols_pca]), index=X_test.index, columns=x_cols_pca)
        X_test = pd.concat([X_test_x_cols, X_test[RE_cols + aux_cols]], axis=1)
    
    # split train to train and eval
    X_train_new, X_eval_new = train_test_split(X_train, test_size=0.1)

    # get dictionaries
    M = 10
    train_data_dict, eval_data_dict, test_data_dict = process_data_for_svgpvae(
        X_train_new, X_test, X_eval_new, x_cols, aux_cols, RE_cols, M, add_train_index_aux=True, sort_train=True)
    train_ids_mask = get_train_ids_mask(train_data_dict, aux_cols)
    
    # run GPPVAE
    X_reconstructed_te, n_epochs = run_experiment_GPPVAE(train_data_dict, eval_data_dict, test_data_dict, train_ids_mask,
        d, q, batch_size, epochs, patience, n_neurons, dropout, activation, verbose, elbo_arg='GPVAE_Casale',
        M = M, RE_cols=RE_cols, aux_cols=aux_cols)
    
    if scale:
        X_reconstructed_te = scaler.inverse_transform(X_reconstructed_te)
    
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    return X_reconstructed_te, [None, none_sigmas, none_sigmas_spatial], n_epochs

def get_train_ids_mask(train_data_dict, aux_cols):
    unique_aux_values = train_data_dict['aux_X'].groupby(aux_cols, as_index=False).size()[aux_cols].values
    train_aux = [train_data_dict['aux_X'][train_data_dict['aux_X']['z0'] == x][aux_cols].values for x in np.sort(np.unique(train_data_dict['aux_X']['z0']))]
    train_ids_mask = np.array([np.isclose(x, y).sum() > 0 for y in train_aux for x in unique_aux_values])
    return train_ids_mask

def process_data_for_svgpvae(X_train, X_test, X_eval, x_cols, aux_cols, RE_cols, M, shuffle=False, add_train_index_aux=False, sort_train=False):
    # What is objects data (on which SVGPVAE perform PCA to get more auxiliary data)?
    # for a single categorical: data on q clusters
    # for spatial data: data on q locations
    # for longitudinal: data on q subjects
    # but notice in all of SVGPVAE examples Y, an image, is predicted based on X, an image PCA-ed (+ "angle info")!
    # in simulations we do not have features regarding q clusters/locations/subjects
    # in real data we might
    # but either way these should be concatenated to data *in all time/location/cluster dependent features* which comprise our X
    # then GROUPED and AVERAGED BY subject/location/cluster, then concatenated back to "angle info", before performing PCA, 
    # otherwise information in aux_X is missing to get good reconstructions in data_Y
    # furthermore perform PCA on training data only! (in all SVGPVAE MNIST example the test data is a missing angle (image))
    # then eval/test should be projected
    train_data_dict, pca, scaler = process_X_for_svgpvae(X_train, x_cols, RE_cols, aux_cols, M = M, shuffle=shuffle, add_train_index_aux=add_train_index_aux, sort_train=sort_train)
    eval_data_dict, _, _ = process_X_for_svgpvae(X_eval, x_cols, RE_cols, aux_cols, pca, scaler, M, shuffle)
    test_data_dict, _, _ = process_X_for_svgpvae(X_test, x_cols, RE_cols, aux_cols, pca, scaler, M, shuffle)
    return train_data_dict, eval_data_dict, test_data_dict

def process_X_for_svgpvae(X, x_cols, RE_cols, aux_cols, pca=None, scaler=None, M=None, shuffle=False, add_train_index_aux=False, sort_train=False):
    X_grouped = X.groupby(RE_cols)[x_cols].mean()
    X_index = X_grouped.index
    if M is None:
        M = int(X.shape[1] * 0.1)
    if pca is None: # training data, perform PCA
        pca = PCA(n_components=M)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_grouped.drop(aux_cols, axis=1))
        X_trans = pd.DataFrame(pca.fit_transform(X_scaled), index = X_index, columns=['PC' + str(i) for i in range(M)])
    else:
        X_scaled = scaler.transform(X_grouped.drop(aux_cols, axis=1))
        X_trans = pd.DataFrame(pca.transform(X_scaled), index = X_index)
    X_aux = X[RE_cols + aux_cols].join(X_trans, on = RE_cols)
    data_Y = X[x_cols].drop(aux_cols, axis=1)
    if shuffle:
        perm = np.random.permutation(X_aux.shape[0])
        data_Y = data_Y.iloc[perm]
        X_aux = X_aux.iloc[perm]
    if sort_train:
        X_aux.index = np.arange(X_aux.shape[0])
        data_Y.index = np.arange(X_aux.shape[0])
        X_aux = X_aux.sort_values(RE_cols + aux_cols)
        data_Y['id'] = np.arange(data_Y.shape[0])
        data_Y['id'] = data_Y['id'].map({l: i for i, l in enumerate(X_aux.index)})
        data_Y = data_Y.sort_values('id')
        data_Y = data_Y.drop('id', axis=1)
    if add_train_index_aux:
        X_aux.insert(loc=0, column='id', value=np.arange(X_aux.shape[0]))
    data_dict = {
        'data_Y': data_Y,
        'aux_X': X_aux
    }
    return data_dict, pca, scaler


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
    elif dr_type == 'svgpvae':
        X_reconstructed_te, sigmas, n_epochs = run_svgpvae(
            X_train, X_test, x_cols, RE_cols_prefix, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, mode, batch_size,
            epochs, patience, n_neurons, dropout, activation, verbose)
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
