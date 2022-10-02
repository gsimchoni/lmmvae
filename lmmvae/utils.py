from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


def get_RE_cols_by_prefix(df, prefix, mode, pca_type='lmmvae'):
    RE_cols = list(df.columns[df.columns.str.startswith(prefix)])
    if mode == 'longitudinal' and pca_type not in ['gppvae', 'svgpvae']:
        RE_cols.append('t')
    return  RE_cols


def get_aux_cols(mode):
    if mode == 'categorical':
        return []
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2', 'spatial_and_categorical']:
        return ['D1', 'D2']
    if mode == 'longitudinal':
        return ['t']
    raise ValueError(f'mode {mode} not recognized')


def verify_q(qs, q_spatial, mode):
    if mode in ['categorical', 'longitudinal']:
        return qs[0]
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2']:
        return q_spatial
    if mode == 'spatial_and_categorical':
        if len(qs) > 1:
            raise ValueError(f'SVGPVAE is not implemented in spatial mode for more than 1 categorical features')
        return qs[0]
    raise ValueError(f'mode {mode} not recognized')


def process_one_hot_encoding(X_train, X_test, x_cols, RE_cols_prefix, mode):
    RE_cols = get_RE_cols_by_prefix(X_train, RE_cols_prefix, mode)
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
        X_trans = pd.DataFrame(pca.transform(X_scaled), index = X_index, columns=['PC' + str(i) for i in range(M)])
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
    if mode in ['categorical', 'spatial_and_categorical']:
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
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2', 'spatial_and_categorical']:
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
        fs = np.random.poisson(params['n_per_cat'], q_spatial) + 1
        fs_sum = fs.sum()
        ps = fs / fs_sum
        ns = np.random.multinomial(n, ps)
        Z_idx = np.repeat(range(q_spatial), ns)
        if mode == 'spatial_and_categorical':
            Z_idx_list.insert(0, Z_idx)
            B_list.insert(0, B)
        else:
            Z_idx_list = [Z_idx]
            B_list = [B]
        Z = get_dummies(Z_idx, q_spatial)
        X += Z @ B
        coords_df = pd.DataFrame(coords[Z_idx])
        co_cols = ['D1', 'D2']
        coords_df.columns = co_cols
    if mode == 'longitudinal':
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
    if mode in ['spatial', 'spatial_fit_categorical', 'spatial2', 'spatial_and_categorical']:
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


def verify_M(x_cols, M, RE_cols, aux_cols):
    n_cols_for_pca = len([col for col in x_cols if col not in RE_cols + aux_cols])
    if n_cols_for_pca < M:
        if n_cols_for_pca < 10:
            M = n_cols_for_pca
        else:
            M = int(0.1 * n_cols_for_pca)
        raise Warning(f'M cannot be larger than no. of features in Y, choosing M = {M} instead')
    return M


def verify_RE_cols(mode, RE_cols):
    if mode == 'spatial_and_categorical' and len(RE_cols) > 1:
        return RE_cols[1:]