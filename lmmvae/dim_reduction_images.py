import gc
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from lmmvae.utils import DRResult

from lmmvae.vae_images import LMMVAEIMG, VAEIMG, VAEIMGCNN, LMMVAEIMGCNN

def process_one_hot_encoding_numpy(X_train, X_test, Z_train, Z_test):
    X_train_new = pd.DataFrame(X_train, columns=list(
            map(lambda c: 'x' + '_' + str(c), range(X_train.shape[1]))))
    X_test_new = pd.DataFrame(X_test, columns=list(
            map(lambda c: 'x' + '_' + str(c), range(X_test.shape[1]))))
    for RE_col in range(Z_train.shape[1]):
        X_train_ohe = pd.get_dummies(Z_train[:, RE_col])
        X_test_ohe = pd.get_dummies(Z_test[:, RE_col])
        X_test_cols_in_train = list(set(
            X_test_ohe.columns).intersection(X_train_ohe.columns))
        X_train_cols_not_in_test = list(set(
            X_train_ohe.columns).difference(X_test_ohe.columns))
        X_test_comp = pd.DataFrame(np.zeros((Z_test.shape[0], len(X_train_cols_not_in_test))),
                                   columns=X_train_cols_not_in_test, dtype=np.uint8)
        X_test_ohe_comp = pd.concat(
            [X_test_ohe[X_test_cols_in_train], X_test_comp], axis=1)
        X_test_ohe_comp = X_test_ohe_comp[X_train_ohe.columns]
        X_train_ohe.columns = list(
            map(lambda c: 'z' + str(RE_col) + '_' + str(c), X_train_ohe.columns))
        X_test_ohe_comp.columns = list(
            map(lambda c: 'z' + str(RE_col) + '_' + str(c), X_test_ohe_comp.columns))
        X_train_new = pd.concat([X_train_new, X_train_ohe], axis=1)
        X_test_new = pd.concat([X_test_new, X_test_ohe_comp], axis=1)
    return X_train_new, X_test_new

def run_pca_ohe_or_ignore_images(X_train, X_test, Z_train, Z_test, d,
    n_sig2bs, n_sig2bs_spatial, mode, verbose, ignore_RE=False):
    n_test, img_height, img_width, channels = X_test.shape
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    if not ignore_RE:
        X_train, X_test = process_one_hot_encoding_numpy(X_train, X_test, Z_train, Z_test)
    pca = PCA(n_components=d)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_transformed_tr = pca.fit_transform(X_train)
    X_transformed_te = pca.transform(X_test)
    X_reconstructed_te = pca.inverse_transform(X_transformed_te)
    X_reconstructed_te = scaler.inverse_transform(X_reconstructed_te)
    X_reconstructed_te = X_reconstructed_te[:, :(img_height * img_width * channels)].reshape(n_test, img_height, img_width, channels)

    loss_tr, loss_te = -pca.score(X_train), -pca.score(X_test)

    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    none_losses = [None for _ in range(3)]
    losses = [loss_tr] + none_losses + [loss_te] + none_losses
    return X_reconstructed_te, [None, none_sigmas, none_sigmas_spatial], None, losses

def run_vae_images(X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, d, n_sig2bs_spatial,
            batch_size, epochs, patience, n_neurons, dropout, activation,
            mode, n_sig2bs, beta, pred_unknown_clusters, verbose, ignore_RE=False, embed_RE=False, cnn=False,
            is_generator=False, train_generator=None, valid_generator=None, test_generator=None):

    if cnn:
        vae = VAEIMGCNN(img_height, img_width, channels, d, batch_size, epochs, patience, n_neurons, dropout, activation,
            beta, pred_unknown_clusters, embed_RE, qs, verbose)
    else:
        vae = VAEIMG(img_height, img_width, channels, d, batch_size, epochs, patience, n_neurons, dropout, activation,
                beta, pred_unknown_clusters, embed_RE, qs, verbose)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    if is_generator:
        X_transformed_tr = vae.fit_transform_gen(train_generator, valid_generator)
        X_transformed_te = vae.transform_gen(test_generator)
        mse_te = vae.recon_error_on_batches(test_generator, X_transformed_te)
        X_reconstructed_te = mse_te # TODO: terrible code
        losses_tr = vae.evaluate_gen(train_generator)
        losses_te = vae.evaluate_gen(test_generator)
    else:
        X_transformed_tr = vae.fit_transform(X_train, Z_train)
        X_transformed_te = vae.transform(X_test, Z_test)
        X_reconstructed_te = vae.reconstruct(X_transformed_te)
        losses_tr = vae.evaluate(X_train, Z_train)
        losses_te = vae.evaluate(X_test, Z_test)

    losses = list(losses_tr) + [None] + list(losses_te) + [None]

    n_epochs = len(vae.get_history().history['loss'])
    none_sigmas = [None for _ in range(n_sig2bs)]
    none_sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    return X_reconstructed_te, [None, none_sigmas, none_sigmas_spatial], n_epochs, losses

def run_lmmvae_images(X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, q_spatial,
                d, n_sig2bs, n_sig2bs_spatial, re_prior, batch_size, epochs, patience, n_neurons,
                n_neurons_re, dropout, activation, mode, beta, kernel_root, pred_unknown_clusters,
                max_spatial_locs, verbose, U, B_list, cnn=False,
                is_generator=False, train_generator=None, valid_generator=None, test_generator=None, train_RE_inputs=None):
    
    if cnn:
        lmmvae = LMMVAEIMGCNN(mode, img_height, img_width, channels, qs, q_spatial,
                    d, n_sig2bs, re_prior, batch_size, epochs, patience, n_neurons, n_neurons_re,
                    dropout, activation, beta, kernel_root, pred_unknown_clusters, verbose)
    else:
        lmmvae = LMMVAEIMG(mode, img_height, img_width, channels, qs, q_spatial,
                        d, n_sig2bs, re_prior, batch_size, epochs, patience, n_neurons, n_neurons_re,
                        dropout, activation, beta, kernel_root, pred_unknown_clusters, verbose)
    
    if is_generator:
        X_transformed_tr, B_hat_list, sig2bs_hat_list = lmmvae.fit_transform_gen(train_generator, valid_generator,
                                                                                 U, B_list, train_RE_inputs=train_RE_inputs)
        X_transformed_te, _, _ = lmmvae.transform_gen(test_generator, U, B_list)
        mse_te = lmmvae.recon_error_on_batches(test_generator, X_transformed_te, B_hat_list)
        X_reconstructed_te = mse_te # TODO: terrible code
        losses_tr = lmmvae.evaluate_gen(train_generator)
        losses_te = lmmvae.evaluate_gen(test_generator)
    else:
        X_transformed_tr, B_hat_list, sig2bs_hat_list = lmmvae.fit_transform(X_train, Z_train, U, B_list)
        X_transformed_te, _, _ = lmmvae.transform(X_test, Z_test, U, B_list)
        X_reconstructed_te = lmmvae.reconstruct(X_transformed_te, Z_test, B_hat_list)
        losses_tr = lmmvae.evaluate(X_train, Z_train)
        losses_te = lmmvae.evaluate(X_test, Z_test)
    
    losses = list(losses_tr)+ list(losses_te)

    n_epochs = len(lmmvae.get_history().history['loss'])
    sig2bs_mean_est = [np.mean(sig2bs) for sig2bs in sig2bs_hat_list]
    sigmas_spatial = [None for _ in range(n_sig2bs_spatial)]
    # TODO: get rid of this
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial_and_categorical']:
        sigmas_spatial = [sig2bs_mean_est[0], None]
        if mode in ['spatial_fit_categorical', 'spatial_and_categorical'] and len(sig2bs_mean_est) > 1:
            sig2bs_mean_est = sig2bs_mean_est[1:]
        else:
            sig2bs_mean_est = []
    return X_reconstructed_te, [None, sig2bs_mean_est, sigmas_spatial], n_epochs, losses


def run_dim_reduction_images(X_train, X_test, Z_train, Z_test, img_height, img_width, channels, d, dr_type,
            thresh, epochs, qs, q_spatial, n_sig2bs, n_sig2bs_spatial,
            est_cors, batch_size, patience, n_neurons, n_neurons_re, dropout,
            activation, mode, beta, re_prior, kernel, pred_unknown_clusters,
            max_spatial_locs, time2measure_dict, verbose, U, B_list,
            train_generator=None, valid_generator=None, test_generator=None, train_RE_inputs=None):
    gc.collect()
    start = time.time()
    losses = [None for _ in range(8)]
    if dr_type == 'pca-ignore':
        X_reconstructed_te, sigmas, n_epochs, losses = run_pca_ohe_or_ignore_images(
            X_train, X_test, Z_train, Z_test, d, n_sig2bs, n_sig2bs_spatial, mode, verbose, ignore_RE=True)
    elif dr_type == 'pca-ohe':
        X_reconstructed_te, sigmas, n_epochs, losses = run_pca_ohe_or_ignore_images(
            X_train, X_test, Z_train, Z_test, d, n_sig2bs, n_sig2bs_spatial, mode, verbose, ignore_RE=False)
    elif dr_type == 'vae-ignore':
        X_reconstructed_te, sigmas, n_epochs, losses = run_vae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, d, n_sig2bs_spatial, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, pred_unknown_clusters, verbose, ignore_RE=True)
    elif dr_type == 'vae-ignore-gen':
        X_reconstructed_te, sigmas, n_epochs, losses = run_vae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, d, n_sig2bs_spatial, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, pred_unknown_clusters, verbose, ignore_RE=True,
            is_generator=True, train_generator=train_generator, valid_generator=valid_generator, test_generator=test_generator)
    elif dr_type == 'vae-ignore-cnn':
        X_reconstructed_te, sigmas, n_epochs, losses = run_vae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, d, n_sig2bs_spatial, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, pred_unknown_clusters, verbose, ignore_RE=True, cnn=True)
    elif dr_type == 'vae-ignore-cnn-gen':
        X_reconstructed_te, sigmas, n_epochs, losses = run_vae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, d, n_sig2bs_spatial, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, pred_unknown_clusters, verbose, ignore_RE=True, cnn=True,
            is_generator=True, train_generator=train_generator, valid_generator=valid_generator, test_generator=test_generator)
    elif dr_type == 'vae-embed':
        if mode == 'spatial':
            qs = [q_spatial]
        if mode == 'spatial_and_categorical':
            qs = [q for q in qs] + [q_spatial]
        X_reconstructed_te, sigmas, n_epochs, losses = run_vae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, d, n_sig2bs_spatial, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, pred_unknown_clusters, verbose, embed_RE=True)
    elif dr_type == 'vae-embed-gen':
        if mode == 'spatial':
            qs = [q_spatial]
        if mode == 'spatial_and_categorical':
            qs = [q for q in qs] + [q_spatial]
        X_reconstructed_te, sigmas, n_epochs, losses = run_vae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, d, n_sig2bs_spatial, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, pred_unknown_clusters, verbose, embed_RE=True,
            is_generator=True, train_generator=train_generator, valid_generator=valid_generator, test_generator=test_generator)
    elif dr_type == 'vae-embed-cnn':
        if mode == 'spatial':
            qs = [q_spatial]
        if mode == 'spatial_and_categorical':
            qs = [q for q in qs] + [q_spatial]
        X_reconstructed_te, sigmas, n_epochs, losses = run_vae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, d, n_sig2bs_spatial, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, pred_unknown_clusters, verbose, embed_RE=True, cnn=True)
    elif dr_type == 'vae-embed-cnn-gen':
        if mode == 'spatial':
            qs = [q_spatial]
        if mode == 'spatial_and_categorical':
            qs = [q for q in qs] + [q_spatial]
        X_reconstructed_te, sigmas, n_epochs, losses = run_vae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, d, n_sig2bs_spatial, batch_size,
            epochs, patience, n_neurons, dropout, activation, mode, n_sig2bs, beta, pred_unknown_clusters, verbose, embed_RE=True, cnn=True,
            is_generator=True, train_generator=train_generator, valid_generator=valid_generator, test_generator=test_generator)
    elif dr_type == 'lmmvae':
        X_reconstructed_te, sigmas, n_epochs, losses = run_lmmvae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, re_prior, batch_size,
            epochs, patience, n_neurons, n_neurons_re, dropout, activation, mode, beta, kernel, pred_unknown_clusters,
            max_spatial_locs, verbose, U, B_list)
    elif dr_type == 'lmmvae-gen':
        X_reconstructed_te, sigmas, n_epochs, losses = run_lmmvae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, re_prior, batch_size,
            epochs, patience, n_neurons, n_neurons_re, dropout, activation, mode, beta, kernel, pred_unknown_clusters,
            max_spatial_locs, verbose, U, B_list,
            is_generator=True, train_generator=train_generator, valid_generator=valid_generator,
            test_generator=test_generator, train_RE_inputs = train_RE_inputs)
    elif dr_type == 'lmmvae-cnn':
        X_reconstructed_te, sigmas, n_epochs, losses = run_lmmvae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, re_prior, batch_size,
            epochs, patience, n_neurons, n_neurons_re, dropout, activation, mode, beta, kernel, pred_unknown_clusters,
            max_spatial_locs, verbose, U, B_list, cnn=True)
    elif dr_type == 'lmmvae-cnn-gen':
        X_reconstructed_te, sigmas, n_epochs, losses = run_lmmvae_images(
            X_train, X_test, Z_train, Z_test, img_height, img_width, channels, qs, q_spatial, d, n_sig2bs, n_sig2bs_spatial, re_prior, batch_size,
            epochs, patience, n_neurons, n_neurons_re, dropout, activation, mode, beta, kernel, pred_unknown_clusters,
            max_spatial_locs, verbose, U, B_list, cnn=True,
            is_generator=True, train_generator=train_generator, valid_generator=valid_generator,
            test_generator=test_generator, train_RE_inputs = train_RE_inputs)
    else:
        raise ValueError(f'{dr_type} is an unknown dr_type')
    end = time.time()
    if mode in ['spatial', 'spatial_fit_categorical', 'longitudinal', 'spatial_and_categorical']:
        pass #x_cols = [x_col for x_col in x_cols if x_col not in ['D1', 'D2', 't']]
    if dr_type.endswith('gen'):
        metric_X = X_reconstructed_te
    else:
        try:
            metric_X = np.mean((X_test - X_reconstructed_te)**2)
        except:
            metric_X = np.nan
    none_rhos = [None for _ in range(len(est_cors))]
    return DRResult(metric_X, sigmas, none_rhos, n_epochs, end - start, losses)
