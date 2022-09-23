from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

DRResult = namedtuple(
    'PCAResult', ['metric_X', 'sigmas', 'rhos', 'n_epochs', 'time'])

Data = namedtuple('PCAData', [
    'X_train', 'X_test', 'W', 'U', 'B_list', 'x_cols', 'kernel'
])

DRInput = namedtuple('PCAInput',
    list(Data._fields) + ['mode', 'N', 'p', 'qs', 'd',
    'sig2e', 'sig2bs_means', 'sig2bs_spatial', 'q_spatial',
    'rhos', 'sig2bs_identical', 'beta', 're_prior',
    'k', 'n_sig2bs', 'n_sig2bs_spatial', 'estimated_cors', 'epochs', 'RE_cols_prefix',
    'thresh', 'batch_size', 'patience', 'n_neurons', 'n_neurons_re', 'dropout',
    'activation', 'verbose'])


def get_dummies(vec, vec_max):
    vec_size = vec.size
    Z = sparse.csr_matrix((np.ones(vec_size), (np.arange(vec_size), vec)), shape=(
        vec_size, vec_max), dtype=np.uint8)
    return Z


def get_columns_by_prefix(df, prefix, mode, pca_type='lmmvae'):
    RE_cols = list(df.columns[df.columns.str.startswith(prefix)])
    if mode == 'longitudinal' and pca_type not in ['gppvae', 'svgpvae']:
        RE_cols.append('t')
    return  RE_cols


def process_one_hot_encoding(X_train, X_test, x_cols, RE_cols_prefix, mode):
    RE_cols = get_columns_by_prefix(X_train, RE_cols_prefix, mode)
    X_train_new = X_train[x_cols]
    X_test_new = X_test[x_cols]
    for RE_col in RE_cols:
        X_train_ohe = pd.get_dummies(X_train[RE_col])
        X_test_ohe = pd.get_dummies(X_test[RE_col])
        X_test_cols_in_train = list(set(
            X_test_ohe.columns).intersection(X_train_ohe.columns))
        X_train_cols_not_in_test = list(set(
            X_train_ohe.columns).difference(X_test_ohe.columns))
        X_test_comp = pd.DataFrame(np.zeros((X_test.shape[0], len(X_train_cols_not_in_test))),
                                   columns=X_train_cols_not_in_test, dtype=np.uint8, index=X_test.index)
        X_test_ohe_comp = pd.concat(
            [X_test_ohe[X_test_cols_in_train], X_test_comp], axis=1)
        X_test_ohe_comp = X_test_ohe_comp[X_train_ohe.columns]
        X_train_ohe.columns = list(
            map(lambda c: RE_col + '_' + str(c), X_train_ohe.columns))
        X_test_ohe_comp.columns = list(
            map(lambda c: RE_col + '_' + str(c), X_test_ohe_comp.columns))
        X_train_new = pd.concat([X_train_new, X_train_ohe], axis=1)
        X_test_new = pd.concat([X_test_new, X_test_ohe_comp], axis=1)
    return X_train_new, X_test_new


def get_cov_mat(sig2bs, rhos, est_cors):
    cov_mat = np.zeros((len(sig2bs), len(sig2bs)))
    for k in range(len(sig2bs)):
        for j in range(len(sig2bs)):
            if k == j:
                cov_mat[k, j] = sig2bs[k]
            else:
                rho_symbol = ''.join(map(str, sorted([k, j])))
                if rho_symbol in est_cors:
                    rho = rhos[est_cors.index(rho_symbol)]
                else:
                    rho = 0
                cov_mat[k, j] = rho * np.sqrt(sig2bs[k]) * np.sqrt(sig2bs[j])
    return cov_mat


def generate_data(mode, n, qs, q_spatial, d, sig2e, sig2bs_means, sig2bs_spatial_mean, rhos, sig2bs_identical, params):
    p = params['n_fixed_features']
    W = np.random.normal(size=p * d).reshape(p, d)
    U = np.random.normal(size=n * d).reshape(n, d)
    mu = np.random.uniform(-10, 10, size=p)#np.zeros(p)
    e = np.random.normal(scale=np.sqrt(sig2e), size=n * p).reshape(n, p)
    UW = U @ W.T
    kernel_root = None
    if params['X_non_linear']:
        fU = (U[:,None,:]*W[None,:,:]*np.cos(U[:,None,:]*W[None,:,:])).sum(axis=2)
    else:
        fU = UW
    X = fU + mu + e
    if mode == 'categorical':
        Z_idx_list = []
        B_list = []
        for k, q in enumerate(qs):
            sig2bs_mean = sig2bs_means[k]
            if sig2bs_mean < 1:
                fs_factor = sig2bs_mean
            else:
                fs_factor = 1
            if sig2bs_identical:
                sig2bs = np.repeat(sig2bs_mean, p)
            else:
                sig2bs = (np.random.poisson(sig2bs_mean, p) + 1) * fs_factor
            D = np.diag(sig2bs)
            B = np.random.multivariate_normal(np.zeros(p), D, q)
            B_list.append(B)
            fs = np.random.poisson(params['n_per_cat'], q) + 1
            fs_sum = fs.sum()
            ps = fs / fs_sum
            ns = np.random.multinomial(n, ps)
            Z_idx = np.repeat(range(q), ns)
            Z = get_dummies(Z_idx, q)
            X += Z @ B
            Z_idx_list.append(Z_idx)
    elif mode in ['spatial', 'spatial_fit_categorical', 'spatial2']:
        if sig2bs_spatial_mean[0] < 1:
            fs_factor = sig2bs_spatial_mean[0]
        else:
            fs_factor = 1
        if sig2bs_identical:
            sig2bs_spatial = np.repeat(sig2bs_spatial_mean[0], p)
        else:
            sig2bs_spatial = (np.random.poisson(sig2bs_spatial_mean[0], p) + 1) * fs_factor
        coords = np.stack([np.random.uniform(-10, 10, q_spatial), np.random.uniform(-10, 10, q_spatial)], axis=1)
        # ind = np.lexsort((coords[:, 1], coords[:, 0]))    
        # coords = coords[ind]
        dist_matrix = squareform(pdist(coords)) ** 2
        kernel = np.exp(-dist_matrix / (2 * sig2bs_spatial_mean[1]))
        # b_list = []
        # for k in range(p):
        #     b_k = np.random.multivariate_normal(np.zeros(q_spatial), sig2bs_spatial[k] * kernel, 1)
        #     b_list.append(b_k)
        # B = np.concatenate(b_list, axis=0).T

        D = np.diag(sig2bs_spatial)
        a = np.random.normal(0, 1, q_spatial * p)
        A = a.reshape(q_spatial, p, order='F')
        M = np.zeros((q_spatial, p))
        D_root = np.linalg.cholesky(D)
        try:
            kernel_root = np.linalg.cholesky(kernel)
        except:
            jitter = 1e-05
            kernel_root = np.linalg.cholesky(kernel + jitter * np.eye(kernel.shape[0]))
        B = M + (kernel_root @ A) @ D_root
        B_list = [B]
        fs = np.random.poisson(params['n_per_cat'], q_spatial) + 1
        fs_sum = fs.sum()
        ps = fs / fs_sum
        ns = np.random.multinomial(n, ps)
        Z_idx = np.repeat(range(q_spatial), ns)
        Z_idx_list = [Z_idx]
        Z = get_dummies(Z_idx, q_spatial)
        X += Z @ B
        coords_df = pd.DataFrame(coords[Z_idx])
        co_cols = ['D1', 'D2']
        coords_df.columns = co_cols
    elif mode == 'longitudinal':
        B_list = []
        fs = np.random.poisson(params['n_per_cat'], qs[0]) + 1
        fs_sum = fs.sum()
        ps = fs/fs_sum
        ns = np.random.multinomial(n, ps)
        Z_idx = np.repeat(range(qs[0]), ns)
        Z_idx_list = [Z_idx]
        max_period = np.arange(ns.max())
        t = 6 * sig2e * np.concatenate([max_period[:k] for k in ns]) / max_period[-1] - sig2e * 3
        estimated_cors = params.get('estimated_cors', [])
        cov_mat = get_cov_mat(sig2bs_means, rhos, estimated_cors)
        # if sig2bs_mean < 1:
        #     fs_factor = sig2bs_mean
        # else:
        #     fs_factor = 1
        # if sig2bs_identical:
        #     sig2bs = np.repeat(sig2bs_mean, p)
        # else:
        #     sig2bs = (np.random.poisson(sig2bs_mean, p) + 1) * fs_factor
        K = len(sig2bs_means)
        D = np.diag(np.ones(p))
        a = np.random.normal(0, 1, K * qs[0] * p)
        A = a.reshape(K * qs[0], p, order='F')
        M = np.zeros((K * qs[0], p))
        D_root = np.linalg.cholesky(D)
        try:
            kernel_root = np.linalg.cholesky(np.kron(cov_mat, np.eye(qs[0])))
        except:
            jitter = 1e-05
            kernel_root = np.linalg.cholesky(kernel + jitter * np.eye(kernel.shape[0]))
        B = M + (kernel_root @ A) @ D_root

        # options
        # B = np.random.multivariate_normal(np.zeros(len(sig2bs_means) * qs[0]), np.kron(cov_mat, np.eye(qs[0])), p)
        # b_list = []
        # for i in range(p):
        #     bs = np.random.multivariate_normal(np.zeros(K), cov_mat, qs[0])
        #     b = bs.reshape((K * qs[0], 1), order = 'F')
        #     b_list.append(b)
        # B = np.hstack(b_list)

        Z0 = sparse.csr_matrix(get_dummies(Z_idx, qs[0]))
        Z_list = [Z0]
        for k in range(1, K):
            Z_list.append(sparse.spdiags(t ** k, 0, n, n) @ Z0)
        ZB = sparse.hstack(Z_list) @ B
        X += ZB
        B_list = [B[(i * qs[0]):((i + 1) * qs[0]), :] for i in range(K)]
    df = pd.DataFrame(X)
    x_cols = ['X' + str(i) for i in range(p)]
    df.columns = x_cols
    for k, Z_idx in enumerate(Z_idx_list):
        df['z' + str(k)] = Z_idx
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2']:
        df = pd.concat([df, coords_df], axis=1)
        x_cols.extend(co_cols)
    if mode == 'longitudinal':
        df['t'] = t
        x_cols.append('t')
        pred_future = params.get('longitudinal_predict_future', False)
    else:
        pred_future = False
    test_size = params.get('test_size', 0.2)
    if pred_future:
        # test set is "the future" or those obs with largest t
        df.sort_values('t', inplace=True)
    X_train, X_test = train_test_split(df, test_size=test_size, shuffle=not pred_future)
    # TODO: why is this necessary?
    X_train = X_train.sort_index()
    X_test = X_test.sort_index()
    U_train = U[X_train.index]
    return Data(X_train, X_test, W, U_train, B_list, x_cols, kernel_root)
