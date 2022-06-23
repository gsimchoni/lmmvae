from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split

PCAResult = namedtuple(
    'PCAResult', ['metric', 'sigmas', 'n_epochs', 'time'])

Data = namedtuple('PCAData', [
    'X_train', 'X_test', 'y_train', 'y_test', 'W', 'U', 'B', 'x_cols'
])

PCAInput = namedtuple('PCAInput', list(Data._fields) + ['N', 'p', 'q', 'd',
                                                        'sig2e', 'sig2bs_mean', 'sig2bs_identical', 'k', 'epochs', 'RE_col',
                                                        'thresh', 'batch_size', 'patience', 'n_neurons', 'dropout',
                                                        'activation', 'verbose'])


def get_dummies(vec, vec_max):
    vec_size = vec.size
    Z = sparse.csr_matrix((np.ones(vec_size), (np.arange(vec_size), vec)), shape=(
        vec_size, vec_max), dtype=np.uint8)
    return Z


def process_one_hot_encoding(X_train, X_test, x_cols, RE_col):
    z_cols = [RE_col]
    X_train_new = X_train[x_cols]
    X_test_new = X_test[x_cols]
    for z_col in z_cols:
        X_train_ohe = pd.get_dummies(X_train[z_col])
        X_test_ohe = pd.get_dummies(X_test[z_col])
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
            map(lambda c: z_col + '_' + str(c), X_train_ohe.columns))
        X_test_ohe_comp.columns = list(
            map(lambda c: z_col + '_' + str(c), X_test_ohe_comp.columns))
        X_train_new = pd.concat([X_train_new, X_train_ohe], axis=1)
        X_test_new = pd.concat([X_test_new, X_test_ohe_comp], axis=1)
    return X_train_new, X_test_new


def generate_data(n, qs, d, sig2e, sig2bs_mean, sig2bs_identical, params):
    p = params['n_fixed_features']
    fs_factor = 1
    W = np.random.normal(size=p * d).reshape(p, d)
    U = np.random.normal(size=n * d).reshape(n, d)
    mu = np.random.uniform(-10, 10, size=p)#np.zeros(p)
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
        fU = (U[:,None,:]*W[None,:,:]*np.cos(U[:,None,:]*W[None,:,:])).sum(axis=2)
    else:
        fU = UW
    X = fU + mu + Z @ B + \
        np.random.normal(scale=np.sqrt(sig2e), size=n * p).reshape(n, p)
    df = pd.DataFrame(X)
    x_cols = ['X' + str(i) for i in range(p)]
    df.columns = x_cols
    df['z'] = Z_idx
    y = U @ np.ones(d) + np.random.normal(size=n, scale=1.0)
    df['y'] = y
    test_size = params['test_size'] if 'test_size' in params else 0.2
    X_train, X_test,  y_train, y_test = train_test_split(
        df.drop('y', axis=1), df['y'], test_size=test_size)
    # TODO: why is this necessary?
    X_train.sort_index(inplace=True)
    X_test.sort_index(inplace=True)
    y_train = y_train[X_train.index]
    y_test = y_test[X_test.index]
    U_train = U[X_train.index]
    return Data(X_train, X_test, y_train, y_test, W, U_train, B, x_cols)
