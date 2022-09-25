import random
import numpy as np
import scipy
import pandas as pd
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def tensor_slice(data_dict, batch_size, placeholder):
    data_Y = tf.data.Dataset.from_tensor_slices(data_dict['data_Y'])
    data_X = tf.data.Dataset.from_tensor_slices(data_dict['aux_X'])
    if placeholder:
        batch_size_placeholder = tf.compat.v1.placeholder(dtype=tf.int64, shape=())
    else:
        batch_size_placeholder = batch_size
    data = tf.data.Dataset.zip((data_Y, data_X)).batch(batch_size_placeholder)
    return data, batch_size_placeholder

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


def generate_init_inducing_points_tabular(train_data, RE_cols, aux_cols, n_samp_per_aux=5, nr_aux_units=16, 
    PCA=False, M=None, seed=0, seed_init=0):
    """
    Generate initial inducing points for tabular data.
    For each angle/location/time we sample n object vectors from empirical distribution of PCA embeddings of training data.

    :param n: how many object vectors per each angle/location/time to sample
    :param nr_aux_units: number of angles/locations between [min, max) (e.g. [0, 2*pi)), for 2D locations should be some L**2 where L is int
    :param PCA: whether or not to use PCA initialization
    :param M: dimension of GPLVM vectors (if none, compute them as aux_data.shape[1] - len(aux_cols))
    :param RE_cols: identity columns
    :param aux_cols: aux column(s) from which space to sample (e.g. angle or lon/lat or time)
    """

    random.seed(seed)

    data = train_data['aux_X']
    aux_data = data.drop(RE_cols + aux_cols, axis=1).values
    if M is None:
        M = data.shape[1] - len(RE_cols + aux_cols)
    aux_units_list = []
    n_aux_cols = len(aux_cols)
    if n_aux_cols == 1:
        aux_units = np.linspace(data[aux_cols[0]].min(), data[aux_cols[0]].max(), nr_aux_units + 1)[:-1]
    elif n_aux_cols == 2:
        for aux_col in aux_cols:
            aux_units = np.linspace(data[aux_col].min(), data[aux_col].max(), int(np.sqrt(nr_aux_units)) + 1)[:-1]
            aux_units_list.append(aux_units)
        aux_units = np.stack(aux_units_list, axis=1)
        xx, yy = np.meshgrid(*aux_units_list)
        aux_units = np.array((xx.ravel(), yy.ravel())).T
    else:
        raise ValueError("Sqrt(nr_aux_units) is only for 2D fields")

    inducing_points = []

    if n_samp_per_aux < 1:
        indices = random.choices(list(range(nr_aux_units)), k = int(n_samp_per_aux * nr_aux_units))
        n_samp_per_aux = 1
    else:
        indices = range(nr_aux_units)

    for i in indices:

        # for reproducibility
        seed = seed_init + i

        if PCA:
            obj_vectors = []
            for pca_ax in range(M):
                # sample from empirical dist of PCA embeddings
                obj_vectors.append(scipy.stats.gaussian_kde(aux_data[:, pca_ax]).resample(int(n_samp_per_aux), seed=seed))

            obj_vectors = np.concatenate(tuple(obj_vectors)).T
        else:
            obj_vectors = np.random.normal(0, 1.5, int(n_samp_per_aux) * M).reshape(int(n_samp_per_aux), M)

        if n_aux_cols == 1:
            d = aux_units[i]
        else:
            d = aux_units[i][np.newaxis, :]
        obj_vectors = np.hstack((np.full((int(n_samp_per_aux), n_aux_cols), d), obj_vectors))  # add angle to each inducing point
        inducing_points.append(obj_vectors)

    inducing_points = np.concatenate(tuple(inducing_points))
    id_col = np.array([list(range(len(inducing_points)))]).T
    inducing_points = np.hstack((id_col, inducing_points))
    return inducing_points

def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation (batch, tmax, 2) tf variable
        var1: var  of expectation (batch, tmax, 2) tf variable
        mu2:  mean of integrand (batch, tmax, 2) tf variable
        var2: var of integrand (batch, tmax, 2) tf variable

    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """
    mu2 = tf.cast(mu2, dtype=tf.float64)
    var2 = tf.cast(var2, dtype=tf.float64)

    term0 = 1.8378770664093453  # log(2*pi)
    term1 = tf.math.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy

def parse_opt_regime(arr):
    arr1 = []
    for i in range(len(arr)):
        regime, nr_epochs = arr[i].split("-")
        arr1.append((regime, int(nr_epochs)))
    training_regime = [[regime[0]] * regime[1] for regime in arr1]
    flatten = lambda l: [item for sublist in l for item in sublist]
    training_regime = flatten(training_regime)
    nr_epochs = len(training_regime)
    return nr_epochs, training_regime

# def plot_tabular(arr, recon_arr, title, nr_images=5, seed=0):
#     """

#     :param arr:
#     :param recon_arr:
#     :param title:
#     :param nr_images:
#     :param seed:
#     :return:
#     """
#     random.seed(seed)
#     assert nr_images < 10

#     fig, axs = plt.subplots(nrows=1, ncols=nr_images)
#     plt.suptitle(title)
#     for i in range(nr_images):
#         axs[i].scatter(arr.values[:, i], recon_arr[:, i])
#     # plt.tight_layout()
#     plt.draw()