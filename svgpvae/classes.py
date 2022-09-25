import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tfp.math.psd_kernels

def _add_diagonal_jitter(matrix, jitter=1e-8):
    return tf.linalg.set_diag(matrix, tf.linalg.diag_part(matrix) + jitter)

class TabularVAE:

    dtype = tf.float64

    def __init__(self, p, L, n_neurons, dropout, activation):
        """
        VAE for tabular data.

        :param p:
        :param L:
        """

        self.p = p
        self.L = L

        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(self.p,), dtype=self.dtype))
        if n_neurons is not None and len(n_neurons) > 0:
            self.encoder.add(tf.keras.layers.Dense(n_neurons[0], activation=activation))
            if dropout is not None and len(dropout) > 0:
                self.encoder.add(tf.keras.Dropout(dropout[0]))
            for i in range(1, len(n_neurons) - 1):
                self.encoder.add(tf.keras.layers.Dense(n_neurons[i], activation=activation))
                if dropout is not None and len(dropout) > i:
                    self.encoder.add(tf.keras.Dropout(dropout[i]))
            if len(n_neurons) > 1:
                self.encoder.add(tf.keras.layers.Dense(n_neurons[-1], activation=activation))
        self.encoder.add(tf.keras.layers.Dense(2 * self.L))

        n_neurons_rev = None if n_neurons is None else list(reversed(n_neurons))
        dropout_rev = None if dropout is None else list(reversed(dropout))
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.InputLayer(input_shape=(self.L,), dtype=self.dtype))
        if n_neurons_rev is not None and len(n_neurons_rev) > 0:
            self.decoder.add(tf.keras.layers.Dense(n_neurons_rev[0], activation=activation))
            if dropout_rev is not None and len(dropout_rev) > 0:
                self.decoder.add(tf.keras.Dropout(dropout[0]))
            for i in range(1, len(n_neurons_rev) - 1):
                self.decoder.add(tf.keras.layers.Dense(n_neurons_rev[i], activation=activation))
                if dropout_rev is not None and len(dropout_rev) > i:
                    self.decoder.add(tf.keras.Dropout(dropout_rev[i]))
            if len(n_neurons_rev) > 1:
                self.decoder.add(tf.keras.layers.Dense(n_neurons_rev[-1], activation=activation))
        self.decoder.add(tf.keras.layers.Dense(self.p))

    def encode(self, data_Y):
        """

        :param images:
        :return:
        """

        encodings = self.encoder(data_Y)
        means, vars = encodings[:, :self.L], tf.exp(encodings[:, self.L:])  # encoder outputs \mu and log(\sigma^2)
        return means, vars

    def decode(self, latent_samples):
        """

        :param latent_samples:
        :return:
        """

        recon_data_Y = self.decoder(latent_samples)
        return recon_data_Y


class MainTabularSVGP:

    def __init__(self, titsias, fixed_inducing_points, initial_inducing_points,
                 name, jitter, N_train, dtype, L, K_obj_normalize=False):
        """
        SVGP main class.

        :param titsias: if true we use L_T (Titsias elbo). Else we use L_H (Hensman elbo).
        :param fixed_inducing_points:
        :param initial_inducing_points:
        :param name: name (or index) of the latent channel
        :param jitter: jitter/noise for numerical stability
        :param N_train: number of training datapoints
        :param L: number of latent channels used in SVGPVAE
        :param K_obj_normalize: whether or not to normalize object linear kernel
        """

        self.dtype = dtype
        self.jitter = jitter
        self.titsias = titsias
        self.nr_inducing = len(initial_inducing_points)
        self.N_train = N_train
        self.L = L
        self.K_obj_normalize = K_obj_normalize

        # u (inducing points)
        if fixed_inducing_points:
            self.inducing_index_points = tf.constant(initial_inducing_points, dtype=self.dtype)
        else:
            self.inducing_index_points = tf.Variable(initial_inducing_points, dtype=self.dtype,
                                                     name='Sparse_GP_inducing_points_{}'.format(name))

    def kernel_matrix(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y).

        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """

        raise NotImplementedError()

    def variational_loss(self, x, y, mu_hat, A_hat, noise=None):
        """
        Computes L_H for the data in the current batch.

        :param x: auxiliary data for current batch (batch, 1 + 1 + M)
        :param y: mean vector for current latent channel, output of the encoder network (batch, 1)
        :param noise: variance vector for current latent channel, output of the encoder network (batch, 1)
        :param mu_hat:
        :param A_hat:

        :return: sum_term, KL_term (variational loss = sum_term + KL_term)  (1,)
        """
        b = tf.shape(x)[0]
        m = self.inducing_index_points.get_shape()[0]
        b = tf.cast(b, dtype=self.dtype)
        m = tf.cast(m, dtype=self.dtype)
        noise = tf.cast(noise, dtype=self.dtype)
        y = tf.cast(y, dtype=self.dtype)

        # kernel matrices
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m,m)
        K_mm_inv = tf.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)

        K_nn = self.kernel_matrix(x, x, x_inducing=False, y_inducing=False, diag_only=True)  # (b)

        K_nm = self.kernel_matrix(x, self.inducing_index_points, x_inducing=False)  # (b, m)
        K_mn = tf.transpose(K_nm, perm=[1, 0])  # (m, b)

        if self.titsias:

            cov_mat = tf.linalg.diag(noise) + tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn))
            trace_term = tf.math.reciprocal_no_nan(noise) * (
                        K_nn - tf.linalg.diag_part(tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn))))  # (b)
            cov_mat_inv = tf.linalg.inv(_add_diagonal_jitter(cov_mat, self.jitter))
            cov_mat_chol = tf.linalg.cholesky(_add_diagonal_jitter(cov_mat, self.jitter))
            cov_mat_log_det = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cov_mat_chol)))

            L_2_term = -0.5 * (b * tf.cast(tf.math.log(2 * np.pi), dtype=self.dtype) + cov_mat_log_det +
                               tf.reduce_sum(y * tf.linalg.matvec(cov_mat_inv, y)) +
                               tf.reduce_sum(trace_term))

            return L_2_term, tf.constant(0.0, dtype=self.dtype)

        else:  # Hensman

            # K_nm \cdot K_mm_inv \cdot m,  (b,)
            mean_vector = tf.linalg.matvec(K_nm,
                                           tf.linalg.matvec(K_mm_inv, mu_hat))

            S = A_hat

            # KL term
            K_mm_chol = tf.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
            S_chol = tf.linalg.cholesky(
                _add_diagonal_jitter(A_hat, self.jitter))
            K_mm_log_det = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(K_mm_chol)))
            S_log_det = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(S_chol)))

            KL_term = 0.5 * (K_mm_log_det - S_log_det - m +
                             tf.linalg.trace(tf.matmul(K_mm_inv, A_hat)) +
                             tf.reduce_sum(mu_hat *
                                           tf.linalg.matvec(K_mm_inv, mu_hat)))

            # diag(K_tilde), (b, )
            precision = tf.math.reciprocal_no_nan(noise)

            K_tilde_terms = precision * (K_nn - tf.linalg.diag_part(tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn))))

            # k_i \cdot k_i^T, (b, m, m)
            lambda_mat = tf.matmul(tf.expand_dims(K_nm, axis=2),
                                   tf.transpose(tf.expand_dims(K_nm, axis=2), perm=[0, 2, 1]))

            # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
            lambda_mat = tf.matmul(K_mm_inv, tf.matmul(lambda_mat, K_mm_inv))

            # Trace terms, (b,)
            trace_terms = precision * tf.linalg.trace(tf.matmul(S, lambda_mat))

            # L_3 sum part, (1,)
            L_3_sum_term = -0.5 * (tf.reduce_sum(K_tilde_terms) + tf.reduce_sum(trace_terms) +
                                   tf.reduce_sum(tf.math.log(noise)) + b * tf.cast(tf.math.log(2 * np.pi), dtype=self.dtype) +
                                   tf.reduce_sum(precision * (y - mean_vector) ** 2))

            return L_3_sum_term, KL_term

    def approximate_posterior_params(self, index_points_test, index_points_train=None, y=None, noise=None):
        """
        Computes parameters of q_S.

        :param index_points_test: X_*
        :param index_points_train: X_Train
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP

        :return: posterior mean at index points,
                 (diagonal of) posterior covariance matrix at index points
        """
        noise = tf.cast(noise, dtype=self.dtype)
        y = tf.cast(y, dtype=self.dtype)
        b = tf.cast(tf.shape(index_points_train)[0], dtype=self.dtype)

        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m,m)
        K_mm_inv = tf.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)
        K_xx = self.kernel_matrix(index_points_test, index_points_test, x_inducing=False,
                                  y_inducing=False, diag_only=True)  # (x)
        K_xm = self.kernel_matrix(index_points_test, self.inducing_index_points, x_inducing=False)  # (x, m)
        K_mx = tf.transpose(K_xm, perm=[1, 0])  # (m, x)

        K_nm = self.kernel_matrix(index_points_train, self.inducing_index_points, x_inducing=False)  # (N, m)
        K_mn = tf.transpose(K_nm, perm=[1, 0])  # (m, N)

        sigma_l = K_mm + (self.N_train / b) * tf.matmul(K_mn,
                                                        tf.multiply(K_nm,
                                                                    tf.math.reciprocal_no_nan(noise)[:, tf.newaxis]))
        sigma_l_inv = tf.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))
        mean_vector = (self.N_train / b) * tf.linalg.matvec(K_xm, tf.linalg.matvec(sigma_l_inv,
                                                              tf.linalg.matvec(K_mn, tf.math.reciprocal_no_nan(
                                                                  noise) * y)))

        K_xm_Sigma_l_K_mx = tf.matmul(K_xm, tf.matmul(sigma_l_inv, K_mx))
        B = K_xx + tf.linalg.diag_part(-tf.matmul(K_xm, tf.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_l_K_mx)

        mu_hat = (self.N_train / b) * tf.linalg.matvec(tf.matmul(K_mm, tf.matmul(sigma_l_inv, K_mn)),
                                                       tf.math.reciprocal_no_nan(noise) * y)
        A_hat = tf.matmul(K_mm, tf.matmul(sigma_l_inv, K_mm))

        return mean_vector, B, mu_hat, A_hat

    def mean_vector_bias_analysis(self, index_points, y=None, noise=None):
        """
        Bias analysis (see C.4 in the Supplementary material).

        :param index_points: auxiliary data
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP
        :return:
        """
        noise = tf.cast(noise, dtype=self.dtype)
        y = tf.cast(y, dtype=self.dtype)

        b = tf.cast(tf.shape(index_points)[0], dtype=self.dtype)

        # kernel matrices
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m,m)
        K_bm = self.kernel_matrix(index_points, self.inducing_index_points, x_inducing=False)  # (b, m)
        K_mb = tf.transpose(K_bm, perm=[1, 0])  # (m, b)

        # compute mean vector
        sigma_l = K_mm + (self.N_train / b) * tf.matmul(K_mb,
                                                        tf.matmul(
                                                            tf.linalg.diag(tf.math.reciprocal_no_nan(noise)),
                                                            K_bm))
        sigma_l_inv = tf.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))
        mean_vector = (self.N_train / b) * tf.linalg.matvec(tf.matmul(K_mm, tf.matmul(sigma_l_inv, K_mb)),
                                                            tf.math.reciprocal_no_nan(noise) * y)
        return mean_vector

    def variable_summary(self):
        """
        Returns values of parameters of sparse GP object. For debugging purposes.
        :return:
        """

        raise NotImplementedError()


class TabularDataSVGP(MainTabularSVGP):

    def __init__(self, titsias, fixed_inducing_points, initial_inducing_points, fixed_gp_params,
                 object_vectors_init, name, jitter, N_train, L, K_obj_normalize, RE_cols, aux_cols):
        """
        SVGP class for tabular data.

        :param titsias: if true we use \mathcal{L}_2 (Titsias elbo). Else we use \mathcal{L}_3 (Hensman elbo).
        :param fixed_inducing_points:
        :param initial_inducing_points:
        :param fixed_gp_params:
        :param object_vectors_init: initial value for object vectors (PCA embeddings).
                        If None, object vectors are fixed throughout training. GPLVM
        :param name: name (or index) of the latent channel
        :param jitter: jitter/noise for numerical stability
        :param N_train: number of training datapoints
        :param L: number of latent channels used in SVGPVAE
        :param K_obj_normalize: whether or not to normalize object linear kernel
        """

        super(TabularDataSVGP, self).__init__(titsias=titsias, fixed_inducing_points=fixed_inducing_points,
                                        initial_inducing_points=initial_inducing_points,
                                        name=name, jitter=jitter,
                                        N_train=N_train, dtype=np.float64, L=L,
                                        K_obj_normalize=K_obj_normalize)

        # GP hyperparams
        if fixed_gp_params:
            self.l_GP = tf.constant(1.0, dtype=self.dtype)
            self.amplitude = tf.constant(1.0, dtype=self.dtype)
        else:
            self.l_GP = tf.Variable(initial_value=1.0, name="GP_length_scale_{}".format(name), dtype=self.dtype)
            self.amplitude = tf.Variable(initial_value=1.0, name="GP_amplitude_{}".format(name), dtype=self.dtype)

        # kernels
        self.kernel_view = tfk.ExpSinSquared(amplitude=self.amplitude, length_scale=self.l_GP, period=2*np.pi)
        # self.kernel_view = tfk.ExponentiatedQuadratic(amplitude=self.amplitude, length_scale=self.l_GP)
        self.kernel_object = tfk.Linear()

        # object vectors (GPLVM)
        if object_vectors_init is not None:
            self.object_vectors = tf.Variable(initial_value=object_vectors_init,
                                              name="GP_object_vectors_{}".format(name),
                                              dtype=self.dtype)
        else:
            self.object_vectors = None
        
        # RE/aux colnames
        self.n_RE_cols = len(RE_cols)
        self.n_aux_cols = len(aux_cols)
        assert self.n_RE_cols == 1, "No. of RE cols > 1 not implemented"

    def kernel_matrix(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y). Kernel from Casale's paper is used for rotated MNIST data.

        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points (ugly but solution using tf.shape did not work...)
        :param y_inducing: whether y is a set of inducing points (ugly but solution using tf.shape did not work...)
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """

        # this stays here as a reminder of a nasty, nasty bug...
        # x_inducing = tf.shape(x)[0] == self.nr_inducing
        # y_inducing = tf.shape(y)[0] == self.nr_inducing

        # unpack auxiliary data
        n_RE_cols = self.n_RE_cols
        n_aux_cols = self.n_aux_cols
        loc_object = n_RE_cols + n_aux_cols
        if self.object_vectors is None:
            x_view, x_object = x[:, n_RE_cols:loc_object], x[:, loc_object:]
            y_view, y_object = y[:, n_RE_cols:loc_object], y[:, loc_object:]
            if n_aux_cols > 1:
                x_view = tf.expand_dims(x_view, axis=1)
                y_view = tf.expand_dims(y_view, axis=1)
        else:
            x_view, y_view = x[:, n_RE_cols:loc_object], y[:, n_RE_cols:loc_object]
            if x_inducing:
                x_object = x[:, loc_object:]
            else:
                x_object = tf.gather(self.object_vectors, tf.cast(x[:, n_RE_cols - 1], dtype=tf.int64))
            if y_inducing:
                y_object = y[:, loc_object:]
            else:
                y_object = tf.gather(self.object_vectors, tf.cast(y[:, n_RE_cols - 1], dtype=tf.int64))

        # compute kernel matrix
        if diag_only:
            view_matrix = self.kernel_view.apply(x_view, y_view)
        else:
            view_matrix = self.kernel_view.matrix(x_view, y_view)

        if diag_only:
            object_matrix = self.kernel_object.apply(x_object, y_object)
            if self.K_obj_normalize:
                obj_norm = tf.math.reduce_euclidean_norm(x_object, axis=1) * tf.math.reduce_euclidean_norm(y_object, axis=1)
                object_matrix = object_matrix / obj_norm
        else:
            object_matrix = self.kernel_object.matrix(x_object, y_object)
            if self.K_obj_normalize:  # normalize object matrix
                obj_norm = 1 / tf.matmul(tf.math.reduce_euclidean_norm(x_object, axis=1, keepdims=True),
                                         tf.transpose(tf.math.reduce_euclidean_norm(y_object, axis=1, keepdims=True),
                                                      perm=[1, 0]))
                object_matrix = object_matrix * obj_norm

        return view_matrix * object_matrix

    def variable_summary(self):
        """
        Returns values of parameters of sparse GP object. For debugging purposes.
        :return:
        """

        return self.l_GP, self.amplitude, self.object_vectors, self.inducing_index_points

