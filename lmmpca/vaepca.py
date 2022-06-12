import tensorflow.keras.backend as K
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, Layer, Reshape, Dot
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.models import Model

from lmmpca.utils import get_dummies


def get_indices(N, Z_idx, min_Z):
        return tf.stack([tf.range(N, dtype=tf.int64), Z_idx - min_Z], axis=1)

def get_indices_v1(N, Z_idx):
    return tf.stack([tf.range(N, dtype=tf.int64), Z_idx], axis=1)

def getZ(N, Z_idx, min_Z, max_Z):
    Z_idx = K.squeeze(Z_idx, axis=1)
    indices = get_indices(N, Z_idx, min_Z)
    return tf.sparse.to_dense(tf.sparse.SparseTensor(indices, tf.ones(N), (N, max_Z - min_Z + 1)))

def getZ_v1(N, q, Z_idx):
    Z_idx = K.squeeze(Z_idx, axis=1)
    indices = get_indices_v1(N, Z_idx)
    return tf.sparse.to_dense(tf.sparse.SparseTensor(indices, tf.ones(N), (N, q)))


class Sampling(Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(K.shape(log_var)) * K.exp(log_var / 2) + mean


def add_layers_functional(X_input, n_neurons, dropout, activation, input_dim):
    if len(n_neurons) > 0:
        x = Dense(n_neurons[0], input_dim=input_dim,
                  activation=activation)(X_input)
        if dropout is not None and len(dropout) > 0:
            x = Dropout(dropout[0])(x)
        for i in range(1, len(n_neurons) - 1):
            x = Dense(n_neurons[i], activation=activation)(x)
            if dropout is not None and len(dropout) > i:
                x = Dropout(dropout[i])(x)
        if len(n_neurons) > 1:
            x = Dense(n_neurons[-1], activation=activation)(x)
        return x
    return X_input


class VAE:
    """VAE Class

    """

    def __init__(self, p, d, batch_size, epochs, patience, n_neurons,
                 dropout, activation, verbose) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.history = None
        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=self.epochs if patience is None else patience)]
        inputs = Input(shape=p)
        z = add_layers_functional(inputs, n_neurons, dropout, activation, p)
        codings_mean = Dense(d)(z)
        codings_log_var = Dense(d)(z)
        codings = Sampling()([codings_mean, codings_log_var])
        self.variational_encoder = Model(
            inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

        decoder_inputs = Input(shape=d)
        n_neurons_rev = None if n_neurons is None else list(reversed(n_neurons))
        dropout_rev = None if dropout is None else list(reversed(dropout))
        x = add_layers_functional(decoder_inputs, n_neurons_rev, dropout_rev, activation, d)
        outputs = Dense(p)(x)
        self.variational_decoder = Model(
            inputs=[decoder_inputs], outputs=[outputs])

        _, _, codings = self.variational_encoder(inputs)
        reconstructions = self.variational_decoder(codings)
        self.variational_ae = Model(inputs=[inputs], outputs=[reconstructions])

        # this is the KL loss, we can either subclass our own VAE class, define the
        # squared loss then do "total_loss = reconstruction_loss + kl_loss"
        # as in Keras docs: https://keras.io/examples/generative/vae/
        # Or we can compile the model with loss='mse' and use its add_loss() method like here
        kl_loss = -0.5 * K.sum(
            1 + codings_log_var -
            K.exp(codings_log_var) - K.square(codings_mean),
            axis=-1)
        self.variational_ae.add_loss(K.mean(kl_loss) / float(p))
        self.variational_ae.compile(loss='mse', optimizer='adam')

    def _fit(self, X):
        self.history = self.variational_ae.fit(X, X, epochs=self.epochs,
                                               callbacks=self.callbacks, batch_size=self.batch_size,
                                               validation_split=0.1, verbose=self.verbose)

    def fit(self, X):
        self._fit(X)
        return self

    def _transform(self, X):
        _, _, X_transformed = self.variational_encoder.predict(X)
        return X_transformed

    def transform(self, X):
        check_is_fitted(self, 'history')
        return self._transform(X)

    def fit_transform(self, X):
        self._fit(X)
        return self._transform(X)

    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history


class LMMVAE:
    """LMMVAE Class

    """

    def __init__(self, p, x_cols, RE_col, q, d, re_prior, batch_size, epochs, patience, n_neurons,
                 dropout, activation, verbose) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.history = None
        self.re_prior = tf.constant(re_prior)
        self.x_cols = x_cols
        self.RE_col = RE_col
        self.p = p
        self.q = q
        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=self.epochs if patience is None else patience)]
        X_input = Input(shape=p)
        z = add_layers_functional(X_input, n_neurons, dropout, activation, p)
        codings_mean = Dense(d)(z)
        codings_log_var = Dense(d)(z)
        re_codings_mean = Dense(p * q)(z)
        re_codings_log_var = Dense(p)(z)
        re_codings_log_var = tf.tile(re_codings_log_var, [1, q])
        codings = Sampling()([codings_mean, codings_log_var])
        re_codings = Sampling()([re_codings_mean, re_codings_log_var])
        self.variational_encoder = Model(
            inputs=[X_input], outputs=[
                codings_mean, codings_log_var, codings,
                re_codings_mean, re_codings_log_var, re_codings
            ]
        )

        decoder_inputs = Input(shape=d)
        decoder_re_inputs = Input(shape=q*p)
        Z_input = Input(shape=(1,), dtype=tf.int64)
        n_neurons_rev = None if n_neurons is None else list(reversed(n_neurons))
        dropout_rev = None if dropout is None else list(reversed(dropout))
        x = add_layers_functional(decoder_inputs, n_neurons_rev, dropout_rev, activation, d)
        decoder_output = Dense(p)(x)
        Z = CategoryEncoding(max_tokens=q, output_mode='binary')(Z_input)
        # Z = CategoryEncoding(num_tokens=q, output_mode='one_hot')(Z_input) # TF2.8+
        # Z = getZ_v1(self.batch_size, q, Z_input)
        B = Reshape((q, p))(decoder_re_inputs)
        ZB = Dot(axes=1)([Z, B])
        outputs = decoder_output + ZB
        self.variational_decoder = Model(
            inputs=[decoder_inputs, decoder_re_inputs, Z_input], outputs=[outputs])

        _, _, codings, _, _, re_codings = self.variational_encoder(X_input)
        reconstructions = self.variational_decoder([codings, re_codings, Z_input])
        self.variational_ae = Model(inputs=[X_input, Z_input], outputs=[reconstructions])

        kl_loss = -0.5 * K.sum(
            1 + codings_log_var -
            K.exp(codings_log_var) - K.square(codings_mean),
            axis=-1)
        re_kl_loss = -0.5 * K.sum(
            1 + re_codings_log_var - self.re_prior -
            K.exp(re_codings_log_var - self.re_prior) - K.square(re_codings_mean) * K.exp(-self.re_prior),
            axis=-1)
        self.variational_ae.add_loss(K.mean(kl_loss) / float(p))
        self.variational_ae.add_loss(K.mean(re_kl_loss) / float(p))
        self.variational_ae.compile(loss='mse', optimizer='adam')

    def _fit(self, X):
        X, Z = X[self.x_cols].copy(), X[self.RE_col].copy()
        self.history = self.variational_ae.fit([X, Z], X, epochs=self.epochs,
            callbacks=self.callbacks, batch_size=self.batch_size, validation_split=0.1,
            verbose=self.verbose)

    def fit(self, X):
        self._fit(X)
        return self

    def _transform(self, X, predict_B):
        X, Z = X[self.x_cols].copy(), X[self.RE_col].copy()
        Z = get_dummies(Z, self.q)
        if predict_B:
            _, _, X_transformed, _, _, B = self.variational_encoder.predict(X)
            self.B = B.mean(axis=0).reshape((self.q, self.p))
        _, _, X_transformed, _, _, _ = self.variational_encoder.predict(X - Z @ self.B)
        return X_transformed

    def transform(self, X, predict_B=True):
        check_is_fitted(self, 'history')
        return self._transform(X, predict_B)

    def fit_transform(self, X, predict_B=True):
        self._fit(X)
        return self._transform(X, predict_B)

    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history
