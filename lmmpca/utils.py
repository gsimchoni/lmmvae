from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

PCAResult = namedtuple(
    'PCAResult', ['metric_y', 'metric_X', 'sigmas', 'n_epochs', 'time'])

Data = namedtuple('PCAData', [
    'X_train', 'X_test', 'y_train', 'y_test', 'W', 'U', 'B_list', 'x_cols', 'kernel'
])

PCAInput = namedtuple('PCAInput', list(Data._fields) + ['mode', 'N', 'p', 'qs', 'd',
                                                        'sig2e', 'sig2bs_means', 'sig2bs_spatial', 'q_spatial',
                                                        'sig2bs_identical', 'beta', 're_prior',
                                                        'k', 'n_sig2bs_spatial', 'epochs', 'RE_cols_prefix',
                                                        'thresh', 'batch_size', 'patience', 'n_neurons', 'dropout',
                                                        'activation', 'verbose'])


def get_dummies(vec, vec_max):
    vec_size = vec.size
    Z = sparse.csr_matrix((np.ones(vec_size), (np.arange(vec_size), vec)), shape=(
        vec_size, vec_max), dtype=np.uint8)
    return Z


def get_columns_by_prefix(df, prefix):
    return df.columns[df.columns.str.startswith(prefix)]


def process_one_hot_encoding(X_train, X_test, x_cols, RE_cols_prefix):
    RE_cols = get_columns_by_prefix(X_train, RE_cols_prefix)
    X_train_new = X_train[x_cols]
    X_test_new = X_test[x_cols]
    for RE_col in RE_cols:
        X_train_ohe = pd.get_dummies(X_train[RE_col])
        X_test_ohe = pd.get_dummies(X_test[RE_col])
        X_test_cols_in_train = set(
            X_test_ohe.columns).intersection(X_train_ohe.columns)
        X_train_cols_not_in_test = set(
            X_train_ohe.columns).difference(X_test_ohe.columns)
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


def generate_data(mode, n, qs, q_spatial, d, sig2e, sig2bs_means, sig2bs_spatial_mean, sig2bs_identical, params):
    p = params['n_fixed_features']
    W = np.random.normal(size=p * d).reshape(p, d)
    U = np.random.normal(size=n * d).reshape(n, d)
    mu = np.random.uniform(-10, 10, size=p)#np.zeros(p)
    e = np.random.normal(scale=np.sqrt(sig2e), size=n * p).reshape(n, p)
    UW = U @ W.T
    kernel = None
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
    if mode in ['spatial', 'spatial_fit_categorical']:
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
        b_list = []
        for k in range(p):
            b_k = np.random.multivariate_normal(np.zeros(q_spatial), sig2bs_spatial[k] * kernel, 1)
            b_list.append(b_k)
        B = np.concatenate(b_list, axis=0).T
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
    df = pd.DataFrame(X)
    x_cols = ['X' + str(i) for i in range(p)]
    df.columns = x_cols
    for k, Z_idx in enumerate(Z_idx_list):
        df['z' + str(k)] = Z_idx
    if mode in ['spatial', 'spatial_fit_categorical']:
        df = pd.concat([df, coords_df], axis=1)
        x_cols.extend(co_cols)
    y = U @ np.ones(d) + np.random.normal(size=n, scale=1.0)
    df['y'] = y
    test_size = params.get('test_size', 0.2)
    X_train, X_test,  y_train, y_test = train_test_split(
        df.drop('y', axis=1), df['y'], test_size=test_size)
    # TODO: why is this necessary?
    X_train.sort_index(inplace=True)
    X_test.sort_index(inplace=True)
    y_train = y_train[X_train.index]
    y_test = y_test[X_test.index]
    U_train = U[X_train.index]
    return Data(X_train, X_test, y_train, y_test, W, U_train, B_list, x_cols, kernel)
