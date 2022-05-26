from collections import namedtuple

import numpy as np
import scipy.sparse as sparse

PCAResult = namedtuple(
    'PCAResult', ['metric', 'sigmas', 'n_epochs', 'time'])

Data = namedtuple('PCAData', [
    'X_train', 'X_test', 'y_train', 'y_test', 'Z_train', 'Z_test', 'X_big_train', 'Z_big_train',
    'X_mean_train', 'X_big_mean_train', 'n_train', 'ns_train', 'U', 'U_train', 'U_test'
])

PCAInput = namedtuple('PCAInput', list(Data._fields) + ['N', 'p', 'q', 'd',
    'sig2e', 'sig2bs_mean', 'sig2bs_identical', 'k',
    'thresh', 'verbose'])


def get_dummies(vec, vec_max):
    vec_size = vec.size
    Z = sparse.csr_matrix((np.ones(vec_size), (np.arange(vec_size), vec)), shape=(
        vec_size, vec_max), dtype=np.uint8)
    return Z


def generate_data(n, qs, d, sig2e, sig2bs_mean, sig2bs_identical, params):
    p = params['n_fixed_features']
    fs_factor = 1
    tr_p = 1 - params['test_size'] if 'test_size' in params else 0.8
    W = np.random.normal(size=p * d).reshape(p, d)
    U = np.random.normal(size=n * d).reshape(n, d)
    mu = np.random.uniform(-10, 10, size=p)
    if sig2bs_identical:
        sig2bs = np.repeat(sig2bs_mean, p)
    else:
        sig2bs = (np.random.poisson(sig2bs_mean, p) + 1) * fs_factor
    D = np.diag(sig2bs)
    B = np.random.multivariate_normal(np.zeros(p), D, qs[0])
    fs = np.random.poisson(params['n_per_cat'], qs[0]) + 1
    fs_sum = fs.sum()
    ps = fs / fs_sum
    ns = np.random.multinomial(n, ps)
    Z_idx = np.repeat(range(qs[0]), ns)
    Z = get_dummies(Z_idx, qs[0])
    UW = U @ W.T
    if params['X_non_linear']:
        fU = UW * np.cos(UW)
        if d > 1:
            fU += 2 * U[:, 0] * U[:, 1]
    else:
        fU = UW
    X = fU + mu + Z @ B + \
        np.random.normal(scale=np.sqrt(sig2e), size=n * p).reshape(n, p)
    n_train = int(tr_p * n)
    train_ids = np.sort(np.random.choice(
        np.arange(n), size=n_train, replace=False))
    test_ids = np.delete(np.arange(n), train_ids)
    X_train = X[train_ids, :]
    X_test = X[test_ids, :]
    Z_idx_train = Z_idx[train_ids]
    ns_train = np.array([np.sum(Z_idx_train == j) for j in range(qs[0])])
    Z_train = Z[train_ids, :]
    Z_test = Z[test_ids, :]
    U_train = U[train_ids, :]
    U_test = U[test_ids, :]
    Z_big_train = sparse.kron(Z_train, sparse.eye(p))
    X_big_train = X_train.reshape(-1)
    X_mean_train = X_train.mean(axis=0)
    X_big_mean_train = np.tile(X_mean_train, n_train)
    y = U @ np.ones(d) + np.random.normal(size=n, scale = 1.0)
    y_train = y[train_ids]
    y_test = y[test_ids]
    return Data(X_train, X_test, y_train, y_test, Z_train, Z_test, X_big_train, Z_big_train,
                X_mean_train, X_big_mean_train, n_train, ns_train, U, U_train, U_test)
