import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

from lmmpca.utils import PCAResult

def reg_pca_ohe_or_ignore(X_train, X_test, y_train, y_test, Z_train, Z_test, d, verbose, ignore_RE=False):
    pca = PCA(n_components=d)
    if ignore_RE:
        X_data_tr = X_train
        X_data_te = X_test
    else:
        X_data_tr = np.hstack((X_train, Z_train.toarray()))
        X_data_te = np.hstack((X_test, Z_test.toarray()))
    
    scaler = StandardScaler()
    scaler.fit(X_data_tr)
    X_data_tr = scaler.transform(X_data_tr)
    X_data_te = scaler.transform(X_data_te)

    pca.fit(X_data_tr)
    W = pca.components_.T

    lm_fit = LinearRegression().fit(X_data_tr @ W, y_train)
    y_pred = lm_fit.predict(X_data_te @ W)
    return y_pred, [None, None], None


def reg_lmmpca(X_train, X_test, y_train, y_test, thresh, verbose):
    print(1)
    pass


def reg_pca(X_train, X_test, y_train, y_test,  Z_train, Z_test, d, pca_type, thresh, verbose):
    start = time.time()
    if pca_type == 'ohe':
        y_pred, sigmas, n_epochs = reg_pca_ohe_or_ignore(X_train, X_test, y_train, y_test,
            Z_train, Z_test, d, verbose)
    elif pca_type == 'lmmpca':
        y_pred, sigmas, n_epochs = reg_lmmpca(X_train, X_test, y_train, y_test,
            thresh, verbose)
    elif pca_type == 'ignore':
        y_pred, sigmas, n_epochs = reg_pca_ohe_or_ignore(
            X_train, X_test, y_train, y_test, Z_train, Z_test, d, verbose, ignore_RE=True)
    else:
        raise ValueError(f'{pca_type} is an unknown pca_type')
    end = time.time()
    metric = mse(y_test, y_pred)
    return PCAResult(metric, sigmas, n_epochs, end - start)
