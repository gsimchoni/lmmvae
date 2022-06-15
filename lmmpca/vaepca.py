import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.utils.validation import check_is_fitted
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, Layer
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer


class Sampling(Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(K.shape(log_var)) * K.exp(log_var / 2) + mean


class Orthogonal(Regularizer):
    def __init__(self, encoding_dim, reg_weight = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.reg_weight = reg_weight
        self.axis = axis

    def __call__(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            m = K.dot(K.transpose(w), w) - tf.eye(self.encoding_dim)
            return self.reg_weight * K.sqrt(K.sum(K.square(K.abs(m))))
        else:
            m = K.sum(w ** 2) - 1.
            return m


def add_layers_functional(X_input, n_neurons, dropout, activation, input_dim):
    if n_neurons is not None and len(n_neurons) > 0:
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
        K.clear_session()
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
        Z_input = Input(shape=(1,), dtype=tf.int64)
        Z = CategoryEncoding(max_tokens=q, output_mode='binary')(Z_input)
        # Z = CategoryEncoding(num_tokens=q, output_mode='one_hot')(Z_input) # TF2.8+
        z = Concatenate()([X_input, Z])
        z = add_layers_functional(z, n_neurons, dropout, activation, p)
        # codings_mean = Dense(d, kernel_regularizer=Orthogonal(d))(z)
        codings_mean = Dense(d)(z)
        codings_log_var = Dense(d)(z)
        re_codings_mean = Dense(p)(z)
        re_codings_log_var = Dense(p)(z)
        codings = Sampling()([codings_mean, codings_log_var])
        re_codings = Sampling()([re_codings_mean, re_codings_log_var])
        self.variational_encoder = Model(
            inputs=[X_input, Z_input], outputs=[codings, re_codings]
        )

        decoder_inputs = Input(shape=d)
        decoder_re_inputs = Input(shape=p)
        
        n_neurons_rev = None if n_neurons is None else list(reversed(n_neurons))
        dropout_rev = None if dropout is None else list(reversed(dropout))
        x = add_layers_functional(decoder_inputs, n_neurons_rev, dropout_rev, activation, d)
        decoder_output = Dense(p)(x)
        B = K.dot(K.transpose(Z), decoder_re_inputs) / K.sum(Z)
        ZB = K.dot(Z, B)
        outputs = decoder_output + ZB
        self.variational_decoder = Model(
            inputs=[decoder_inputs, decoder_re_inputs, Z_input], outputs=[outputs])

        codings, re_codings = self.variational_encoder([X_input, Z_input])
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
        self.variational_ae.add_loss(MeanSquaredError()(X_input, reconstructions))
        self.variational_ae.compile(optimizer='adam')

    def _fit(self, X):
        X, Z = X[self.x_cols].copy(), X[self.RE_col].copy()
        self.history = self.variational_ae.fit([X, Z], X, epochs=self.epochs,
            callbacks=self.callbacks, batch_size=self.batch_size, validation_split=0.1,
            verbose=self.verbose)

    def fit(self, X):
        self._fit(X)
        return self

    def _transform(self, X, U, B):
        X, Z = X[self.x_cols].copy(), X[self.RE_col].copy()
        X_transformed, B_hat = self.variational_encoder.predict([X, Z])
        B, B_hat = self.extract_Bs_to_compare(B, Z, B_hat)
        return X_transformed
    
    def extract_Bs_to_compare(self, B, Z, B_hat):
        B_df = pd.DataFrame(B_hat)
        B_df['z'] = Z.values
        B_df2 = B_df.groupby('z')[B_df.columns[:self.p]].mean()
        idx_not_in_B = np.setdiff1d(np.arange(self.q), B_df2.index)
        B2 = np.delete(B, idx_not_in_B, axis=0)
        return B2, B_df2.values

    def transform(self, X, U, B):
        check_is_fitted(self, 'history')
        return self._transform(X, U, B)

    def fit_transform(self, X, U, B):
        self._fit(X)
        return self._transform(X, U, B)

    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history
