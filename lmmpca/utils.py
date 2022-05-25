from collections import namedtuple

import numpy as np
import scipy.sparse as sparse

PCAResult = namedtuple(
    'PCAResult', ['metric', 'sigmas', 'rhos', 'weibull', 'n_epochs', 'time'])

PCAInput = namedtuple('PCAInput', ['X_train', 'X_test', 'y_train', 'y_test', 'x_cols',
                                   'N', 'qs', 'sig2e', 'p_censor', 'sig2bs', 'rhos', 'sig2bs_spatial', 'q_spatial',
                                   'k', 'batch', 'epochs', 'patience',
                                   'Z_non_linear', 'Z_embed_dim_pct', 'mode', 'n_sig2bs', 'n_sig2bs_spatial', 'estimated_cors',
                                   'dist_matrix', 'time2measure_dict', 'verbose', 'n_neurons', 'dropout', 'activation',
                                   'spatial_embed_neurons', 'log_params',
                                   'weibull_lambda', 'weibull_nu', 'resolution', 'shuffle'])


def get_dummies(vec, vec_max):
    vec_size = vec.size
    Z = sparse.csr_matrix((np.ones(vec_size), (np.arange(vec_size), vec)), shape=(
        vec_size, vec_max), dtype=np.uint8)
    return Z


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def generate_data(n, p, q, d, sig2e, sig2bs_mean, sig2bs_identical, tr_p=0.8, fs_mean=30, fs_factor=1):
    W = np.random.normal(size=p * d).reshape(p, d)
    U = np.random.normal(size=n * d).reshape(n, d)
    mu = np.random.uniform(-10, 10, size=p)
    if sig2bs_identical:
        sig2bs = np.repeat(sig2bs_mean, p)
    else:
        sig2bs = (np.random.poisson(sig2bs_mean, p) + 1) * fs_factor
    D = np.diag(sig2bs)
    B = np.random.multivariate_normal(np.zeros(p), D, q)
    fs = np.random.poisson(fs_mean, q) + 1
    fs_sum = fs.sum()
    ps = fs/fs_sum
    ns = np.random.multinomial(n, ps)
    Z_idx = np.repeat(range(q), ns)
    Z = get_dummies(Z_idx, q)
    X = U @ W.T + mu + Z @ B + \
        np.random.normal(scale=np.sqrt(sig2e), size=n * p).reshape(n, p)
    n_train = int(tr_p * n)
    train = np.sort(np.random.choice(
        np.arange(n), size=n_train, replace=False))
    test = np.delete(np.arange(n), train)
    X_train = X[train, :]
    X_test = X[test, :]
    Z_idx_train = Z_idx[train]
    ns_train = np.array([np.sum(Z_idx_train == j) for j in range(q)])
    Z_train = Z[train, :]
    Z_test = Z[test, :]
    U_train = U[train, :]
    U_test = U[test, :]
    Z_big_train = sparse.kron(Z_train, sparse.eye(p))
    X_big_train = X_train.reshape(-1)
    X_mean_train = X_train.mean(axis=0)
    X_big_mean_train = np.tile(X_mean_train, n_train)
    return X_train, X_test, Z_train, Z_test, X_big_train, Z_big_train, X_mean_train, X_big_mean_train, n_train, ns_train, U, U_train, U_test, train, test
