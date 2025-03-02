import gc

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow.keras.backend as K
from packaging import version
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, Layer, Embedding, Reshape, Masking, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer

from lmmvae.utils import get_dummies


class Sampling(Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(K.shape(log_var)) * K.exp(log_var / 2) + mean


class SamplingFull(Layer):
    def call(self, inputs, kernel_root):
        mean, log_var = inputs
        return K.random_normal(K.shape(log_var)) * K.exp(log_var / 2) @ kernel_root + mean


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
                 dropout, activation, beta, pred_unknown_clusters, embed_RE,
                 qs, x_cols, RE_cols, verbose) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.pred_unknown_clusters = pred_unknown_clusters
        self.embed_RE = embed_RE
        self.qs = qs
        self.RE_cols = RE_cols
        self.x_cols = x_cols
        self.Z_embed_dim = 10
        self.history = None
        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=self.epochs if patience is None else patience)]
        X_input = Input(shape=p)
        Z_inputs = []
        concat = X_input
        input_dim = p
        if self.embed_RE:
            embeds = []
            for q in self.qs:
                Z_input = Input(shape=(1,))
                embed = Embedding(q, self.Z_embed_dim, input_length=1)(Z_input)
                embed = Reshape(target_shape=(self.Z_embed_dim,))(embed)
                Z_inputs.append(Z_input)
                embeds.append(embed)
            concat = Concatenate()([X_input] + embeds)
            input_dim = p + self.Z_embed_dim * len(self.qs)
        z = add_layers_functional(concat, n_neurons, dropout, activation, input_dim)
        codings_mean = Dense(d)(z)
        codings_log_var = Dense(d)(z)
        codings = Sampling()([codings_mean, codings_log_var])
        self.variational_encoder = Model(
            inputs=[X_input] + Z_inputs, outputs=[codings_mean, codings_log_var, codings])

        decoder_inputs = Input(shape=d)
        n_neurons_rev = None if n_neurons is None else list(reversed(n_neurons))
        dropout_rev = None if dropout is None else list(reversed(dropout))
        x = add_layers_functional(decoder_inputs, n_neurons_rev, dropout_rev, activation, d)
        outputs = Dense(p)(x)
        self.variational_decoder = Model(
            inputs=[decoder_inputs], outputs=[outputs])

        _, _, codings = self.variational_encoder([X_input] + Z_inputs)
        reconstructions = self.variational_decoder(codings)
        self.variational_ae = Model(inputs=[X_input] + Z_inputs, outputs=[reconstructions])

        # this is the KL loss, we can either subclass our own VAE class, define the
        # squared loss then do "total_loss = reconstruction_loss + kl_loss"
        # as in Keras docs: https://keras.io/examples/generative/vae/
        # Or we can compile the model with loss='mse' and use its add_loss() method like here
        kl_loss = -0.5 * K.sum(
            1 + codings_log_var -
            K.exp(codings_log_var) - K.square(codings_mean),
            axis=-1)
        kl_loss = K.mean(kl_loss)
        self.variational_ae.add_loss(beta * kl_loss)
        recon_loss = MeanSquaredError()(X_input, reconstructions) * p
        self.variational_ae.add_loss(recon_loss)
        self.variational_ae.add_metric(recon_loss, name='recon_loss')
        self.variational_ae.add_metric(kl_loss, name='kl_loss')
        self.variational_ae.compile(optimizer='adam')

    def _fit(self, X):
        if self.embed_RE:
            X_inputs = [X[self.x_cols].copy()]
            Z_inputs = [X[RE_col].copy() for RE_col in self.RE_cols]
        else:
            X_inputs = [X]
            Z_inputs = []
        self.history = self.variational_ae.fit(X_inputs + Z_inputs, X_inputs, epochs=self.epochs,
                                               callbacks=self.callbacks, batch_size=self.batch_size,
                                               validation_split=0.1, verbose=self.verbose)
        gc.collect()

    def fit(self, X):
        self._fit(X)
        return self

    def _get_input(self, X):
        if self.embed_RE:
            X_inputs = [X[self.x_cols].copy()]
            Z_inputs = [X[RE_col].copy() for RE_col in self.RE_cols]
        else:
            X_inputs = [X]
            Z_inputs = []
        return X_inputs, Z_inputs
    
    def _transform(self, X):
        X_inputs, Z_inputs = self._get_input(X)
        _, _, X_transformed = self.variational_encoder.predict(X_inputs + Z_inputs, verbose=0)
        return X_transformed

    def transform(self, X):
        check_is_fitted(self, 'history')
        return self._transform(X)

    def fit_transform(self, X):
        self._fit(X)
        return self._transform(X)
    
    def reconstruct(self, X_transformed):
        X_reconstructed = self.variational_decoder.predict([X_transformed], verbose=0)
        return X_reconstructed

    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history
    
    def evaluate(self, X):
        X_inputs, Z_inputs = self._get_input(X)
        total_loss, recon_loss, kl_loss = self.variational_ae.evaluate(X_inputs + Z_inputs, verbose=0)
        return total_loss, recon_loss, kl_loss


class VRAE:
    """VRAE Class

    """

    def __init__(self, p, nt, d, batch_size, epochs, patience, n_lstm_cells,
                 dropout, activation, beta, pred_unknown_clusters, verbose) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.pred_unknown_clusters = pred_unknown_clusters
        self.history = None
        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=self.epochs if patience is None else patience)]
        X_input = Input(shape=(nt, p))
        z = Masking(mask_value=.0)(X_input)
        z, h, c = LSTM(n_lstm_cells, return_state=True)(z)
        codings_mean = Dense(d)(z)
        codings_log_var = Dense(d)(z)
        codings = Sampling()([codings_mean, codings_log_var])
        self.variational_encoder = Model(
            inputs=[X_input], outputs=[codings_mean, codings_log_var, codings, h, c])

        decoder_inputs = Input(shape=d)
        latent_repeat = RepeatVector(nt)(decoder_inputs)
        h = Input(shape=(n_lstm_cells, ), name='encoder_state_h')
        c = Input(shape=(n_lstm_cells, ), name='encoder_state_c')
        decoder_lstm = LSTM(n_lstm_cells, return_sequences=True)
        decoder_dense = TimeDistributed(Dense(p))
        x = decoder_lstm(latent_repeat, initial_state=[h, c])

        outputs = decoder_dense(x)
        self.internal_decoder = Model(
            inputs=[decoder_inputs, h, c], outputs=[outputs])

        _, _, codings, h_decoded, c_decoded = self.variational_encoder([X_input])
        reconstructions = self.internal_decoder([codings, h_decoded, c_decoded])
        self.variational_ae = Model(inputs=[X_input], outputs=[reconstructions])

        kl_loss = -0.5 * K.sum(
            1 + codings_log_var -
            K.exp(codings_log_var) - K.square(codings_mean),
            axis=-1)
        self.variational_ae.add_loss(beta * K.mean(kl_loss))
        self.variational_ae.add_loss(MeanSquaredError()(X_input, reconstructions))
        self.variational_ae.compile(optimizer='adam')

    def _fit(self, X):
        self.history = self.variational_ae.fit(X, X, epochs=self.epochs,
                                               callbacks=self.callbacks, batch_size=self.batch_size,
                                               validation_split=0.1, verbose=self.verbose)
        gc.collect()

    def fit(self, X):
        self._fit(X)
        return self

    def _transform(self, X):
        _, _, X_transformed, _, _ = self.variational_encoder.predict(X, verbose=0)
        return X_transformed

    def transform(self, X):
        check_is_fitted(self, 'history')
        return self._transform(X)

    def fit_transform(self, X):
        self._fit(X)
        return self._transform(X)
    
    def reconstruct(self, X_transformed):
        X_reconstructed = self.variational_decoder.predict([X_transformed], verbose=0)
        return X_reconstructed
    
    def predict(self, X):
        X_reconstructed = self.variational_ae.predict([X], verbose=0)
        return X_reconstructed

    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history


class LMMVAE:
    """LMMVAE Class

    """

    def __init__(self, mode, p, x_cols, RE_cols, qs, q_spatial, d, n_sig2bs,
                re_prior, batch_size, epochs, patience, n_neurons, n_neurons_re,
                dropout, activation, beta, kernel_root, pred_unknown_clusters, verbose) -> None:
        super().__init__()
        K.clear_session()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.history = None
        self.re_prior = tf.constant(np.log(re_prior, dtype=np.float32))
        self.x_cols = x_cols
        self.RE_cols = RE_cols
        self.p = p
        self.mode = mode
        self.pred_unknown_clusters = pred_unknown_clusters
        self.n_sig2bs = n_sig2bs
        self.qs = qs
        if self.mode in ['spatial_fit_categorical', 'spatial_and_categorical']:
            self.qs = [q_spatial] + list(qs)
        if self.mode == 'spatial':
            self.qs = [q_spatial]
        if self.mode in ['spatial', 'spatial_and_categorical']:
            self.kernel_root = tf.constant(kernel_root, dtype=tf.float32)
        self.n_RE_inputs = len(self.qs) if mode in ['categorical', 'spatial_fit_categorical', 'spatial_and_categorical'] else 1
        self.n_RE_outputs = self.n_sig2bs if mode == 'longitudinal' else self.n_RE_inputs
        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=self.epochs if patience is None else patience)]
        X_input = Input(shape=p)
        Z_inputs = []
        Z_mats = []
        for i in range(self.n_RE_inputs):
            Z_input = Input(shape=(1,), dtype=tf.int64)
            Z_inputs.append(Z_input)
            if version.parse(tf.__version__) >= version.parse('2.8'):
                Z = CategoryEncoding(num_tokens=self.qs[i], output_mode='one_hot')(Z_input)
            else:
                Z = CategoryEncoding(max_tokens=self.qs[i], output_mode='binary')(Z_input)
            Z_mats.append(Z)
        if mode == 'longitudinal':
            t_input = Input(shape=(1,))
            Z_inputs.append(t_input)
            for k in range(1, self.n_sig2bs):
                T = tf.linalg.tensor_diag(K.squeeze(t_input, axis=1) ** k)
                Z = K.dot(T, Z_mats[0])
                Z_mats.append(Z)
        z1 = add_layers_functional(X_input, n_neurons, dropout, activation, p)
        # codings_mean = Dense(d, kernel_regularizer=Orthogonal(d))(z1)
        codings_mean = Dense(d)(z1)
        codings_log_var = Dense(d)(z1)
        # to have a single encoder with double output delete this row (z2) and replace z2 with z1 below
        z2 = add_layers_functional(X_input, n_neurons_re, dropout, activation, p)
        re_codings_mean_list = []
        re_codings_log_var_list = []
        re_codings_list = []
        for _ in range(self.n_RE_outputs):
            re_codings_mean = Dense(p)(z2)
            re_codings_mean_list.append(re_codings_mean)
            re_codings_log_var = Dense(p)(z2)
            re_codings_log_var_list.append(re_codings_log_var)
            re_codings = Sampling()([re_codings_mean, re_codings_log_var])
            re_codings_list.append(re_codings)

        codings = Sampling()([codings_mean, codings_log_var])
        self.variational_encoder = Model(
                inputs=[X_input], outputs=[codings] + re_codings_list
            )

        decoder_inputs = Input(shape=d)
        decoder_re_inputs_list = []
        for _ in range(self.n_RE_outputs):
            decoder_re_inputs = Input(shape=p)
            decoder_re_inputs_list.append(decoder_re_inputs)
        
        n_neurons_rev = None if n_neurons is None else list(reversed(n_neurons))
        dropout_rev = None if dropout is None else list(reversed(dropout))
        x = add_layers_functional(decoder_inputs, n_neurons_rev, dropout_rev, activation, d)
        decoder_output = Dense(p)(x)
        outputs = decoder_output
        for i in range(self.n_RE_outputs):
            Z = Z_mats[i]
            decoder_re_inputs = decoder_re_inputs_list[i]
            q_ind = 0 if self.mode == 'longitudinal' else i
            Z0 = Z_mats[q_ind]
            B = tf.math.divide_no_nan(K.dot(K.transpose(Z0), decoder_re_inputs), K.reshape(K.sum(Z0, axis=0), (self.qs[q_ind], 1)))
            if mode == 'spatial' or (mode == 'spatial_and_categorical' and i == 0):
                B = self.kernel_root @ B
            ZB = K.dot(Z, B)
            outputs += ZB
        self.variational_decoder = Model(
                inputs=[decoder_inputs] + decoder_re_inputs_list + Z_inputs, outputs=[outputs])
        self.variational_decoder_no_re = Model(
            inputs=[decoder_inputs], outputs=[decoder_output])

        encoder_outputs = self.variational_encoder([X_input])
        codings = encoder_outputs[0]
        re_codings_list = encoder_outputs[1:]
        reconstructions = self.variational_decoder([codings] + re_codings_list + Z_inputs)
        self.variational_ae = Model(inputs=[X_input] + Z_inputs, outputs=[reconstructions])


        kl_loss = -0.5 * K.sum(
            1 + codings_log_var -
            K.exp(codings_log_var) - K.square(codings_mean),
            axis=-1)
        kl_loss = K.mean(kl_loss)
        for i in range(self.n_RE_outputs):
            re_codings_mean = re_codings_mean_list[i]
            re_codings_log_var = re_codings_log_var_list[i]
            re_kl_loss_i = -0.5 * K.sum(
                1 + re_codings_log_var - self.re_prior -
                K.exp(re_codings_log_var - self.re_prior) - K.square(re_codings_mean) * K.exp(-self.re_prior),
                axis=-1)
            re_kl_loss_i = K.mean(re_kl_loss_i)
            if i == 0:
                re_kl_loss = re_kl_loss_i
            else:
                re_kl_loss += re_kl_loss_i
        self.variational_ae.add_loss(beta * kl_loss)
        self.variational_ae.add_loss(beta * re_kl_loss)
        recon_loss = MeanSquaredError()(X_input, reconstructions) * p
        self.variational_ae.add_loss(recon_loss)
        self.variational_ae.add_metric(recon_loss, name='recon_loss')
        self.variational_ae.add_metric(kl_loss, name='kl_loss')
        self.variational_ae.add_metric(re_kl_loss, name='re_kl_loss')
        self.variational_ae.compile(optimizer='adam')

    def _fit(self, X):
        X_input = X[self.x_cols].copy()
        Z_inputs = [X[RE_col].copy() for RE_col in self.RE_cols]
        if self.pred_unknown_clusters:
            Z_inputs_train, Z_inputs_valid = train_test_split(Z_inputs[0].unique(), test_size=0.1)
            Z_inputs_train = [Z_input[Z_inputs[0].isin(Z_inputs_train)] for Z_input in Z_inputs]
            Z_inputs_valid =  [Z_input[Z_inputs[0].isin(Z_inputs_valid)] for Z_input in Z_inputs]
            X_input_train = X_input.loc[Z_inputs_train[0].index]
            X_input_valid = X_input.loc[Z_inputs_valid[0].index]
            X_train = X.loc[Z_inputs_train[0].index]
            X_valid = X.loc[Z_inputs_valid[0].index]
            self.history = self.variational_ae.fit([X_input_train] + Z_inputs_train, X_train, epochs=self.epochs,
                callbacks=self.callbacks, batch_size=self.batch_size, validation_data=([X_input_valid] + Z_inputs_valid, X_valid),
                verbose=self.verbose)
        else:
            self.history = self.variational_ae.fit([X_input] + Z_inputs, X, epochs=self.epochs,
                callbacks=self.callbacks, batch_size=self.batch_size, validation_split=0.1,
                verbose=self.verbose)
        gc.collect()

    def fit(self, X):
        self._fit(X)
        return self

    def _get_input(self, X):
        X_inputs = [X[self.x_cols].copy()]
        Z_inputs = [X[RE_col].copy() for RE_col in self.RE_cols]
        return X_inputs, Z_inputs
    
    def _transform(self, X, U, B_list, extract_B):
        X_inputs, Z_inputs = self._get_input(X)
        encoder_output = self.variational_encoder.predict(X_inputs, verbose=0)
        X_transformed = encoder_output[0]
        B_hat_list = encoder_output[1:]
        if extract_B:
            B_hat_list_processed = self.extract_Bs_to_compare(Z_inputs, B_hat_list)
            sig2bs_hat_list = [B_hat_list_processed[i].var(axis=0) for i in range(len(B_hat_list_processed))]
            return X_transformed, B_hat_list_processed, sig2bs_hat_list
        else:
            return X_transformed, None, None
    
    def extract_Bs_to_compare(self, Z_inputs, B_hat_list):
        B_df2_list = []
        for i in range(self.n_RE_outputs):
            B_df = pd.DataFrame(B_hat_list[i])
            q_ind = 0 if self.mode == 'longitudinal' else i
            B_df['z'] = Z_inputs[q_ind].values
            B_df2 = B_df.groupby('z')[B_df.columns[:self.p]].mean()
            B_df2 = B_df2.reindex(range(self.qs[q_ind]), fill_value=0)
            if self.mode == 'spatial' or (self.mode == 'spatial_and_categorical' and i == 0):
                B_df2 = pd.DataFrame(self.kernel_root.numpy() @ B_df2.values)
            B_df2_list.append(B_df2)
        return B_df2_list

    def transform(self, X, U, B_list, extract_B=False):
        check_is_fitted(self, 'history')
        return self._transform(X, U, B_list, extract_B)

    def fit_transform(self, X, U, B_list, reconstruct_B=True):
        self._fit(X)
        return self._transform(X, U, B_list, reconstruct_B)

    def reconstruct(self, X_transformed, Z_idxs, B_list):
        X_reconstructed = self.variational_decoder_no_re.predict([X_transformed], verbose=0)
        if self.mode == 'longitudinal':
            Z0 = sparse.csr_matrix(get_dummies(Z_idxs.iloc[:, 0], self.qs[0]))
            t = Z_idxs.iloc[:, 1]
            n = X_transformed.shape[0]
            for k in range(self.n_sig2bs):
                Z = sparse.spdiags(t ** k, 0, n, n) @ Z0
                X_reconstructed += Z @ B_list[k]
        else:
            for i in range(Z_idxs.shape[1]):
                Z = get_dummies(Z_idxs.iloc[:, i], self.qs[i])
                X_reconstructed += Z @ B_list[i]
        return X_reconstructed
    
    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history
    
    def evaluate(self, X):
        X_inputs, Z_inputs = self._get_input(X)
        total_loss, recon_loss, kl_loss, re_kl_loss = \
            self.variational_ae.evaluate(X_inputs + Z_inputs, verbose=0)
        return total_loss, recon_loss, kl_loss, re_kl_loss
