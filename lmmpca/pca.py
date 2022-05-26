import time

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from lmmpca.utils import get_dummies


class LMMPCA(BaseEstimator, TransformerMixin):
    """LMMPCA Class

    """

    def __init__(self, n_components, max_it, tolerance, cardinality, verbose) -> None:
        super().__init__()
        self.d = n_components
        self.max_it = max_it
        self.thresh = tolerance
        self.q = cardinality
        self.verbose = verbose

    def fit(self, X, y=None, RE_col='z'):
        # ns
        self._fit(X, RE_col)
        return self

    def _fit(self, X, RE_col):
        x_cols = [col for col in X.columns if col != RE_col]
        n = X[x_cols].shape[0]
        p = X[x_cols].shape[1]
        W_c, U_c, B_c, sig2e_c, sig2bs_c = self._generate_candidate_estimates(
            X[x_cols], n, p, self.d, self.q)
        ns = np.array([np.sum(X[RE_col] == j) for j in range(self.q)])
        Z = get_dummies(X[RE_col], self.q)
        X_big = X[x_cols].values.reshape(-1)
        Z_big = sparse.kron(Z, sparse.eye(p))
        X_mean = X[x_cols].values.mean(axis=0)
        X_big_mean = np.tile(X_mean, n)
        W_c, B_c, U_c, ll, time_it, sig2e_est, sig2bs_est, n_iter = \
            self._em_algo(self.max_it, X[x_cols].values, Z, X_big, Z_big, X_mean,
                          X_big_mean, W_c, U_c, B_c, sig2e_c, sig2bs_c, n,
                          ns, self.thresh, p, self.q, self.d, self.verbose)
        self.components_ = W_c
        self.mu = X_mean
        self.B = B_c
        self.sig2e_est = sig2e_est
        self.sig2bs_est = sig2bs_est
        self.n_iter = n_iter

    def transform(self, X, y=None, RE_col='z'):
        check_is_fitted(self, 'components_')
        return self._transform(X, RE_col)

    def fit_transform(self, X, y=None, RE_col='z'):
        self._fit(X, RE_col)
        return self._transform(X, RE_col)

    def _transform(self, X, RE_col):
        x_cols = [col for col in X.columns if col != RE_col]
        Z = get_dummies(X[RE_col], self.q)
        X = X[x_cols].copy()
        return (X - self.mu - Z @ self.B) @ self.components_

    def _generate_candidate_estimates(self, X, n, p, d, q, init_pca=False):
        sig2bs_c = np.ones(p)
        sig2e_c = 1
        if init_pca:
            pca = PCA(n_components=d)
            pca.fit(X)
            W_c = pca.components_.T
        else:
            W_c = np.random.normal(size=p * d).reshape(p, d)
        B_c = np.random.normal(size=q * p).reshape(q, p)
        U_c = np.random.normal(size=n * d).reshape(n, d)
        return W_c, U_c, B_c, sig2e_c, sig2bs_c

    def _em_algo(self, max_it, X, Z, X_big, Z_big, X_mean, X_big_mean, W_c, U_c, B_c,
                 sig2e_c, sig2bs_c, n, ns, thresh, p, q, d, verbose):
        X_c = U_c @ W_c.T + X_mean + Z @ B_c
        frob_norm = np.linalg.norm(X - X_c, ord='fro')
        start = time.time()
        for it in range(max_it):
            W_c, B_c, U_c, ll, time_it, sig2e_c, sig2bs_c = self._em_iteration(
                X, X_big, Z_big, X_mean, X_big_mean, W_c, U_c, B_c, sig2e_c,
                sig2bs_c, n, ns, p, q, d)
            X_c = U_c @ W_c.T + X_mean + Z @ B_c
            frob_norm_next = np.linalg.norm(X - X_c, ord='fro')
            if verbose:
                print(
                    f'    em_it: {it}, frob_norm: {frob_norm_next:.2f}, ll: {ll:.2f}, t: {int(time_it)}')
            if frob_norm - frob_norm_next < thresh * frob_norm:
                break
            else:
                frob_norm = frob_norm_next
        end = time.time()
        return W_c, B_c, U_c, ll, end - start, sig2e_c, sig2bs_c, it + 1

    def _em_iteration(self, X, X_big, Z_big, X_mean, X_big_mean, W_c, U_c, B_c, sig2e_c, sig2bs_c, n, ns, p, q, d):
        start = time.time()
        idx = 0
        sum_w_nom = 0
        sum_w_denom = 0
        u_u_c_list = []
        M = W_c.T @ W_c + sig2e_c * np.eye(d)  # d X d
        M_inv = np.linalg.inv(M)
        var_u_c = sig2e_c * M_inv

        for j in range(q):
            if ns[j] > 0:
                for _ in range(ns[j]):
                    u_c = M_inv @ W_c.T @ (X[idx, :] - X_mean - B_c[j, :])
                    U_c[idx, :] = u_c.copy()
                    u_c = u_c[:, None]
                    sum_w_nom = sum_w_nom + \
                        (X[idx, :] - X_mean - B_c[j, :])[:, None] @ u_c.T
                    sum_w_denom = sum_w_denom + u_c @ u_c.T
                    u_u_c = var_u_c + u_c @ u_c.T
                    u_u_c_list.append(u_u_c)
                    idx += 1
        W_c_new = sum_w_nom @ np.linalg.inv(sum_w_denom)  # d X d inversion

        sum_sig2e_c = 0
        idx = 0
        for j in range(q):
            if ns[j] > 0:
                for _ in range(ns[j]):
                    element = np.sum((X[idx, :] - X_mean - B_c[j, :])**2) - \
                        2 * U_c[idx, :].T @ W_c_new.T @ (X[idx, :] - X_mean - B_c[j, :]) + \
                        np.trace(u_u_c_list[idx] @ W_c_new.T @ W_c_new)
                    sum_sig2e_c += element
                    idx += 1
        sig2e_c = sum_sig2e_c / (n * p)
        W_c = W_c_new.copy()

        D_c = np.diag(sig2bs_c)
        D_big_c = sparse.kron(sparse.eye(q), D_c)
        # V_big = Z_big @ D_big_c @ Z_big.T + sig2e_c * sparse.eye(n * p) # np X np
        # V_big_inv = sparse.linalg.inv(sparse.csc_matrix(V_big))
        V_i_inv_list = []
        for i, ns_i in enumerate(ns):
            if ns_i > 0:
                V_i = sparse.kron(np.ones((ns_i, ns_i)),
                                  D_c) + sig2e_c * sparse.eye(ns_i * p)
                V_i_inv = sparse.linalg.inv(sparse.csc_matrix(V_i))
                V_i_inv_list.append(V_i_inv)
        V_big_inv = sparse.block_diag(V_i_inv_list)
        b = D_big_c @ Z_big.T @ V_big_inv @ (X_big -
                                             X_big_mean - (U_c @ W_c.T).reshape(-1))
        B_c = b.reshape(q, p)

        sig2bs_c_new = np.zeros(p)
        for k in range(p):
            idx_k_rows, idx_k_cols = p * \
                np.arange(n) + k, np.repeat(range(q), ns)
            Z_k = sparse.csr_matrix(
                (np.ones(n), (idx_k_rows, idx_k_cols)), shape=(Z_big.shape[0], q))
            b_c = B_c[:, k]
            b_b = b_c.T @ b_c
            Var_b_b = (sig2bs_c[k] * sparse.eye(q) - sig2bs_c[k]
                       ** 2 * Z_k.T @ V_big_inv @ Z_k).diagonal().sum()
            sig2bs_c_new[k] = (b_b + Var_b_b) / q

        sig2bs_c = sig2bs_c_new.copy()

        ll = 0
        end = time.time()
        time_it = end - start
        return W_c, B_c, U_c, ll, time_it, sig2e_c, sig2bs_c
