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
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, Layer, Embedding, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model

from lmmvae.vae import Sampling, add_layers_functional
from lmmvae.utils import get_dummies
from lmmvae.utils_images import custom_generator_fit, custom_generator_predict, divide, get_full_RE_cols_from_generator

def add_layers_functional_conv2d(X_input, n_neurons, dropout, activation, input_dim):
    if n_neurons is not None and len(n_neurons) > 0:
        x = Conv2D(n_neurons[0], 3, input_dim=input_dim,
                  activation=activation, strides=2, padding='same')(X_input)
        if dropout is not None and len(dropout) > 0:
            x = Dropout(dropout[0])(x)
        for i in range(1, len(n_neurons) - 1):
            x = Conv2D(n_neurons[i], 3, activation=activation, strides=2, padding='same')(x)
            if dropout is not None and len(dropout) > i:
                x = Dropout(dropout[i])(x)
        if len(n_neurons) > 1:
            x = Conv2D(n_neurons[-1], 3, activation=activation, strides=2, padding='same')(x)
        x = Flatten()(x)
        return x
    x = Flatten()(X_input)
    return x

def add_layers_functional_conv2d_t(X_input, n_neurons, dropout, activation, input_dim, img_height, img_width, channels):
    img_height_neurons = img_height // (len(n_neurons)**2) # has to divide by 4 for 2 layers
    img_width_neurons = img_width // (len(n_neurons)**2) # has to divide by 4 for 2 layers
    if n_neurons is not None and len(n_neurons) > 0:
        x = Dense(img_height_neurons * img_width_neurons * n_neurons[0],
                  activation=activation,input_dim=input_dim)(X_input)
        x = Reshape((img_height_neurons, img_width_neurons, n_neurons[0]))(x)
        x = Conv2DTranspose(n_neurons[0], 3, activation=activation,
                            strides=2, padding='same')(x)
        if dropout is not None and len(dropout) > 0:
            x = Dropout(dropout[0])(x)
        for i in range(1, len(n_neurons) - 1):
            x = Conv2DTranspose(n_neurons[i], 3, activation=activation,
                                strides=2, padding='same')(x)
            if dropout is not None and len(dropout) > i:
                x = Dropout(dropout[i])(x)
        if len(n_neurons) > 1:
            x = Conv2DTranspose(n_neurons[-1], 3, activation=activation, strides=2, padding='same')(x)
        x = Conv2DTranspose(3, 3, activation=activation, padding='same')(x)
        return x
    x = Dense(img_height * img_width * channels, activation="relu")(X_input)
    x = Conv2DTranspose(3, 3, activation=activation)(x)
    return X_input

class VAEIMG:
    """VAE Class for images

    """

    def __init__(self, img_height, img_width, channels, d, batch_size, epochs, patience, n_neurons,
                 dropout, activation, beta, pred_unknown_clusters, embed_RE,
                 qs, verbose) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.pred_unknown_clusters = pred_unknown_clusters
        self.embed_RE = embed_RE
        self.qs = qs
        self.Z_embed_dim = 10
        self.history = None
        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=self.epochs if patience is None else patience)]
        X_input = Input(shape=(img_height, img_width, channels))
        X_input_flatten = Flatten()(X_input)
        Z_inputs = []
        concat = X_input_flatten
        input_img_dim = img_height * img_width * channels
        input_dim = input_img_dim
        if self.embed_RE:
            embeds = []
            for q in self.qs:
                Z_input = Input(shape=(1,))
                embed = Embedding(q, self.Z_embed_dim, input_length=1)(Z_input)
                embed = Reshape(target_shape=(self.Z_embed_dim,))(embed)
                Z_inputs.append(Z_input)
                embeds.append(embed)
            concat = Concatenate()([X_input_flatten] + embeds)
            input_dim = input_img_dim + self.Z_embed_dim * len(self.qs)
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
        x = Dense(input_img_dim)(x)
        outputs = Reshape((img_height, img_width, channels))(x)
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
        recon_loss = MeanSquaredError()(X_input, reconstructions) * input_img_dim
        self.variational_ae.add_loss(recon_loss)
        self.variational_ae.add_metric(recon_loss, name='recon_loss')
        self.variational_ae.add_metric(kl_loss, name='kl_loss')
        self.variational_ae.compile(optimizer='adam')

    def _fit(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        self.history = self.variational_ae.fit(X_inputs + Z_inputs, X_inputs, epochs=self.epochs,
                                               callbacks=self.callbacks, batch_size=self.batch_size,
                                               validation_split=0.1, verbose=self.verbose)
        gc.collect()
    
    def _fit_gen(self, train_generator, valid_generator):
        steps_train = divide(train_generator.n, train_generator.batch_size)
        steps_valid = divide(valid_generator.n, valid_generator.batch_size)
        train_gen = custom_generator_fit(train_generator, self.epochs, self.embed_RE)
        valid_gen = custom_generator_fit(valid_generator, self.epochs, self.embed_RE)
        self.history = self.variational_ae.fit(train_gen,
                                               validation_data=valid_gen,
                                               epochs=self.epochs, callbacks=self.callbacks,
                                               batch_size=self.batch_size, verbose=self.verbose,
                                               steps_per_epoch = steps_train,
                                               validation_steps = steps_valid)
        gc.collect()

    def fit(self, X, Z):
        self._fit(X, Z)
        return self

    def _get_input(self, X, Z):
        if self.embed_RE:
            X_inputs = [X]
            Z_inputs = [Z[:, RE_col] for RE_col in range(len(self.qs))]
        else:
            X_inputs = [X]
            Z_inputs = []
        return X_inputs, Z_inputs
    
    def _transform(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        _, _, X_transformed = self.variational_encoder.predict(X_inputs + Z_inputs, verbose=0)
        return X_transformed

    def _transform_gen(self, generator):
        generator.reset()
        prev_shuffle_state = generator.shuffle
        generator.shuffle = False
        steps = divide(generator.n, generator.batch_size)
        gen = custom_generator_predict(generator, self.epochs, self.embed_RE)
        _, _, X_transformed = self.variational_encoder.predict(gen, steps=steps, verbose=0)
        generator.shuffle = prev_shuffle_state
        return X_transformed
    
    def transform(self, X, Z):
        check_is_fitted(self, 'history')
        return self._transform(X, Z)

    def transform_gen(self, generator):
        check_is_fitted(self, 'history')
        return self._transform_gen(generator)
    
    def fit_transform(self, X, Z):
        self._fit(X, Z)
        return self._transform(X, Z)
    
    def fit_transform_gen(self, train_generator, valid_generator):
        self._fit_gen(train_generator, valid_generator)
        return self._transform_gen(train_generator)
    
    def reconstruct(self, X_transformed):
        X_reconstructed = self.variational_decoder.predict([X_transformed], verbose=0)
        return X_reconstructed
    
    def recon_error_on_batches(self, generator, X_transformed):
        generator.reset()
        steps = divide(generator.n, generator.batch_size)
        total_recon_err = 0
        for i in range(steps):
            batch, _ = generator.next()
            idx = np.arange(i * generator.batch_size, i * generator.batch_size + batch.shape[0])
            batch_reconstructed = self.variational_decoder.predict_on_batch([X_transformed[idx]])
            total_recon_err += np.sum((batch - batch_reconstructed)**2)
        avg_recon_err = total_recon_err / (generator.n * np.prod(generator.image_shape))
        return avg_recon_err

    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history
    
    def evaluate(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        total_loss, recon_loss, kl_loss = self.variational_ae.evaluate(X_inputs + Z_inputs, verbose=0)
        return total_loss, recon_loss, kl_loss
    
    def evaluate_gen(self, generator):
        steps = divide(generator.n, generator.batch_size)
        gen = custom_generator_fit(generator, self.epochs, self.embed_RE)
        total_loss, recon_loss, kl_loss = self.variational_ae.evaluate(gen, steps=steps, verbose=0)
        return total_loss, recon_loss, kl_loss


class VAEIMGCNN:
    """VAE Class for images with convolutional encoders/decoders

    """

    def __init__(self, img_height, img_width, channels, d, batch_size, epochs, patience, n_neurons,
                 dropout, activation, beta, pred_unknown_clusters, embed_RE,
                 qs, verbose) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.pred_unknown_clusters = pred_unknown_clusters
        self.embed_RE = embed_RE
        self.qs = qs
        self.Z_embed_dim = 10
        self.history = None
        self.callbacks = [EarlyStopping(monitor='val_loss',
                                        patience=self.epochs if patience is None else patience)]
        input_img_dim = img_height * img_width * channels
        X_input = Input(shape=(img_height, img_width, channels))
        Z_inputs = []
        input_dim_conv = (img_height, img_width, channels)
        z_conv = add_layers_functional_conv2d(X_input, n_neurons, dropout, activation, input_dim_conv)
        z = z_conv
        if self.embed_RE:
            embeds = []
            for q in self.qs:
                Z_input = Input(shape=(1,))
                embed = Embedding(q, self.Z_embed_dim, input_length=1)(Z_input)
                embed = Reshape(target_shape=(self.Z_embed_dim,))(embed)
                Z_inputs.append(Z_input)
                embeds.append(embed)
            input_dim = self.Z_embed_dim * len(self.qs)
            z = Concatenate()([z_conv] + embeds)
        codings_mean = Dense(d)(z)
        codings_log_var = Dense(d)(z)
        codings = Sampling()([codings_mean, codings_log_var])
        self.variational_encoder = Model(
            inputs=[X_input] + Z_inputs, outputs=[codings_mean, codings_log_var, codings])

        decoder_inputs = Input(shape=d)
        n_neurons_rev = None if n_neurons is None else list(reversed(n_neurons))
        dropout_rev = None if dropout is None else list(reversed(dropout))
        outputs = add_layers_functional_conv2d_t(decoder_inputs, n_neurons_rev, dropout_rev, activation, d, img_height, img_width, channels)
        # x = Dense(input_img_dim)(x)
        # outputs = Reshape((img_height, img_width, channels))(x)
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
        recon_loss = MeanSquaredError()(X_input, reconstructions) * input_img_dim
        self.variational_ae.add_loss(recon_loss)
        self.variational_ae.add_metric(recon_loss, name='recon_loss')
        self.variational_ae.add_metric(kl_loss, name='kl_loss')
        self.variational_ae.compile(optimizer='adam')

    def _fit(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        self.history = self.variational_ae.fit(X_inputs + Z_inputs, X_inputs, epochs=self.epochs,
                                               callbacks=self.callbacks, batch_size=self.batch_size,
                                               validation_split=0.1, verbose=self.verbose)
        gc.collect()

    def _fit_gen(self, train_generator, valid_generator):
        steps_train = divide(train_generator.n, train_generator.batch_size)
        steps_valid = divide(valid_generator.n, valid_generator.batch_size)
        train_gen = custom_generator_fit(train_generator, self.epochs, self.embed_RE)
        valid_gen = custom_generator_fit(valid_generator, self.epochs, self.embed_RE)
        self.history = self.variational_ae.fit(train_gen,
                                               validation_data=valid_gen,
                                               epochs=self.epochs, callbacks=self.callbacks,
                                               batch_size=self.batch_size, verbose=self.verbose,
                                               steps_per_epoch = steps_train,
                                               validation_steps = steps_valid)
        gc.collect()
    
    def fit(self, X, Z):
        self._fit(X, Z)
        return self

    def _get_input(self, X, Z):
        if self.embed_RE:
            X_inputs = [X]
            Z_inputs = [Z[:, RE_col] for RE_col in range(len(self.qs))]
        else:
            X_inputs = [X]
            Z_inputs = []
        return X_inputs, Z_inputs
    
    def _transform(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        _, _, X_transformed = self.variational_encoder.predict(X_inputs + Z_inputs, verbose=0)
        return X_transformed

    def _transform_gen(self, generator):
        generator.reset()
        prev_shuffle_state = generator.shuffle
        generator.shuffle = False
        steps = divide(generator.n, generator.batch_size)
        gen = custom_generator_predict(generator, self.epochs, self.embed_RE)
        _, _, X_transformed = self.variational_encoder.predict(gen, steps=steps, verbose=0)
        generator.shuffle = prev_shuffle_state
        return X_transformed
    
    def transform(self, X, Z):
        check_is_fitted(self, 'history')
        return self._transform(X, Z)

    def transform_gen(self, generator):
        check_is_fitted(self, 'history')
        return self._transform_gen(generator)
    
    def fit_transform(self, X, Z):
        self._fit(X, Z)
        return self._transform(X, Z)
    
    def fit_transform_gen(self, train_generator, valid_generator):
        self._fit_gen(train_generator, valid_generator)
        return self._transform_gen(train_generator)
    
    def reconstruct(self, X_transformed):
        X_reconstructed = self.variational_decoder.predict([X_transformed], verbose=0)
        return X_reconstructed

    def recon_error_on_batches(self, generator, X_transformed):
        generator.reset()
        steps = divide(generator.n, generator.batch_size)
        total_recon_err = 0
        for i in range(steps):
            batch, _ = generator.next()
            idx = np.arange(i * generator.batch_size, i * generator.batch_size + batch.shape[0])
            batch_reconstructed = self.variational_decoder.predict_on_batch([X_transformed[idx]])
            total_recon_err += np.sum((batch - batch_reconstructed)**2)
        avg_recon_err = total_recon_err / (generator.n * np.prod(generator.image_shape))
        return avg_recon_err
    
    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history
    
    def evaluate(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        total_loss, recon_loss, kl_loss = self.variational_ae.evaluate(X_inputs + Z_inputs, verbose=0)
        return total_loss, recon_loss, kl_loss
    
    def evaluate_gen(self, generator):
        steps = divide(generator.n, generator.batch_size)
        gen = custom_generator_fit(generator, self.epochs, self.embed_RE)
        total_loss, recon_loss, kl_loss = self.variational_ae.evaluate(gen, steps=steps, verbose=0)
        return total_loss, recon_loss, kl_loss


class LMMVAEIMG:
    """LMMVAE Class for images

    """

    def __init__(self, mode, img_height, img_width, channels, qs, q_spatial, d, n_sig2bs,
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
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.input_img_dim = img_height * img_width * channels
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
        X_input = Input(shape=(img_height, img_width, channels))
        X_input_flatten = Flatten()(X_input)
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
        z1 = add_layers_functional(X_input_flatten, n_neurons, dropout, activation, self.input_img_dim)
        # codings_mean = Dense(d, kernel_regularizer=Orthogonal(d))(z1)
        codings_mean = Dense(d)(z1)
        codings_log_var = Dense(d)(z1)
        z2 = add_layers_functional(X_input_flatten, n_neurons_re, dropout, activation, self.input_img_dim)
        re_codings_mean_list = []
        re_codings_log_var_list = []
        re_codings_list = []
        for _ in range(self.n_RE_outputs):
            re_codings_mean = Dense(self.input_img_dim)(z2)
            re_codings_mean_list.append(re_codings_mean)
            re_codings_log_var = Dense(self.input_img_dim)(z2)
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
            decoder_re_inputs = Input(shape=self.input_img_dim)
            decoder_re_inputs_list.append(decoder_re_inputs)
        
        n_neurons_rev = None if n_neurons is None else list(reversed(n_neurons))
        dropout_rev = None if dropout is None else list(reversed(dropout))
        x = add_layers_functional(decoder_inputs, n_neurons_rev, dropout_rev, activation, d)
        x = Dense(self.input_img_dim)(x)
        decoder_output = Reshape((img_height, img_width, channels))(x)
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
            ZB = Reshape((img_height, img_width, channels))(ZB)
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
        recon_loss = MeanSquaredError()(X_input, reconstructions) * self.input_img_dim
        self.variational_ae.add_loss(recon_loss)
        self.variational_ae.add_metric(recon_loss, name='recon_loss')
        self.variational_ae.add_metric(kl_loss, name='kl_loss')
        self.variational_ae.add_metric(re_kl_loss, name='re_kl_loss')
        self.variational_ae.compile(optimizer='adam')

    def _fit(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        if self.pred_unknown_clusters:
            Z_inputs_train, Z_inputs_valid = train_test_split(Z_inputs[0].unique(), test_size=0.1)
            Z_inputs_train = [Z_input[Z_inputs[0].isin(Z_inputs_train)] for Z_input in Z_inputs]
            Z_inputs_valid =  [Z_input[Z_inputs[0].isin(Z_inputs_valid)] for Z_input in Z_inputs]
            X_input_train = X_inputs[0].loc[Z_inputs_train[0].index]
            X_input_valid = X_inputs[0].loc[Z_inputs_valid[0].index]
            X_train = X.loc[Z_inputs_train[0].index]
            X_valid = X.loc[Z_inputs_valid[0].index]
            self.history = self.variational_ae.fit([X_input_train] + Z_inputs_train, X_train, epochs=self.epochs,
                callbacks=self.callbacks, batch_size=self.batch_size, validation_data=([X_input_valid] + Z_inputs_valid, X_valid),
                verbose=self.verbose)
        else:
            self.history = self.variational_ae.fit(X_inputs + Z_inputs, X, epochs=self.epochs,
                callbacks=self.callbacks, batch_size=self.batch_size, validation_split=0.1,
                verbose=self.verbose)
        gc.collect()

    def _fit_gen(self, train_generator, valid_generator):
        steps_train = divide(train_generator.n, train_generator.batch_size)
        steps_valid = divide(valid_generator.n, valid_generator.batch_size)
        train_gen = custom_generator_fit(train_generator, self.epochs, with_RE=True)
        valid_gen = custom_generator_fit(valid_generator, self.epochs, with_RE=True)
        self.history = self.variational_ae.fit(train_gen,
                                               validation_data=valid_gen,
                                               epochs=self.epochs, callbacks=self.callbacks,
                                               batch_size=self.batch_size, verbose=self.verbose,
                                               steps_per_epoch = steps_train,
                                               validation_steps = steps_valid)
        gc.collect()

    def fit(self, X, Z):
        self._fit(X, Z)
        return self

    def _get_input(self, X, Z):
        X_inputs = [X]
        Z_inputs = [Z[:, RE_col] for RE_col in range(len(self.qs))]
        return X_inputs, Z_inputs
    
    def _transform(self, X, Z, U, B_list, extract_B):
        X_inputs, Z_inputs = self._get_input(X, Z)
        encoder_output = self.variational_encoder.predict(X_inputs, verbose=0)
        X_transformed = encoder_output[0]
        B_hat_list = encoder_output[1:]
        if extract_B:
            B_hat_list_processed = self.extract_Bs_to_compare(Z_inputs, B_hat_list)
            sig2bs_hat_list = [B_hat_list_processed[i].var(axis=0) for i in range(len(B_hat_list_processed))]
            return X_transformed, B_hat_list_processed, sig2bs_hat_list
        else:
            return X_transformed, None, None
    
    def _transform_gen(self, generator, U, B_list, extract_B):
        steps = divide(generator.n, generator.batch_size)
        generator.reset()
        prev_shuffle_state = generator.shuffle
        generator.shuffle = False
        gen = custom_generator_predict(generator, self.epochs, with_RE=False)
        encoder_output = self.variational_encoder.predict(gen, steps=steps, verbose=0)
        X_transformed = encoder_output[0]
        B_hat_list = encoder_output[1:]
        if extract_B:
            Z_inputs = get_full_RE_cols_from_generator(generator)
            B_hat_list_processed = self.extract_Bs_to_compare(Z_inputs, B_hat_list)
            sig2bs_hat_list = [B_hat_list_processed[i].var(axis=0) for i in range(len(B_hat_list_processed))]
            generator.shuffle = prev_shuffle_state
            return X_transformed, B_hat_list_processed, sig2bs_hat_list
        else:
            generator.shuffle = prev_shuffle_state
            return X_transformed, None, None
    
    def extract_Bs_to_compare(self, Z_inputs, B_hat_list):
        B_df2_list = []
        for i in range(self.n_RE_outputs):
            B_df = pd.DataFrame(B_hat_list[i])
            q_ind = 0 if self.mode == 'longitudinal' else i
            B_df['z'] = Z_inputs[q_ind]
            B_df2 = B_df.groupby('z')[B_df.columns[:self.input_img_dim]].mean()
            B_df2 = B_df2.reindex(range(self.qs[q_ind]), fill_value=0)
            if self.mode == 'spatial' or (self.mode == 'spatial_and_categorical' and i == 0):
                B_df2 = pd.DataFrame(self.kernel_root.numpy() @ B_df2.values)
            # convert to image dims
            B_df2 = B_df2.values.reshape(B_df2.shape[0], self.img_height, self.img_width, self.channels)
            B_df2_list.append(B_df2)
        return B_df2_list

    def transform(self, X, Z, U, B_list, extract_B=False):
        check_is_fitted(self, 'history')
        return self._transform(X, Z, U, B_list, extract_B)

    def transform_gen(self, generator, U, B_list, extract_B=False):
        check_is_fitted(self, 'history')
        return self._transform_gen(generator, U, B_list, extract_B)
    
    def fit_transform(self, X, Z, U, B_list, reconstruct_B=True):
        self._fit(X, Z)
        return self._transform(X, Z, U, B_list, reconstruct_B)
    
    def fit_transform_gen(self, train_generator, valid_generator, U, B_list, reconstruct_B=True):
        self._fit_gen(train_generator, valid_generator)
        return self._transform_gen(train_generator, U, B_list, reconstruct_B)

    def reconstruct(self, X_transformed, Z_idxs, B_list):
        X_reconstructed = self.variational_decoder_no_re.predict([X_transformed], verbose=0)
        if self.mode == 'longitudinal':
            Z0 = sparse.csr_matrix(get_dummies(Z_idxs[:, 0], self.qs[0]))
            t = Z_idxs[:, 1]
            n = X_transformed.shape[0]
            for k in range(self.n_sig2bs):
                Z = sparse.spdiags(t ** k, 0, n, n) @ Z0
                X_reconstructed += Z @ B_list[k]
        else:
            for i in range(Z_idxs.shape[1]):
                Z = get_dummies(Z_idxs[:, i], self.qs[i])
                X_reconstructed += np.tensordot(Z.toarray(),  B_list[i], axes=[1,0])
        return X_reconstructed
    
    def recon_error_on_batches(self, generator, X_transformed, B_list):
        generator.reset()
        steps = divide(generator.n, generator.batch_size)
        total_recon_err = 0
        for i in range(steps):
            batch, Z_idxs = generator.next()
            idx = np.arange(i * generator.batch_size, i * generator.batch_size + batch.shape[0])
            batch_reconstructed = self.variational_decoder_no_re.predict_on_batch([X_transformed[idx]])
            if len(Z_idxs.shape) == 1:
                Z_idxs = Z_idxs[:, np.newaxis]
            for i in range(Z_idxs.shape[1]):
                Z = get_dummies(Z_idxs[:, i], self.qs[i])
                batch_reconstructed += np.tensordot(Z.toarray(),  B_list[i], axes=[1,0])
            total_recon_err += np.sum((batch - batch_reconstructed)**2)
        avg_recon_err = total_recon_err / (generator.n * np.prod(generator.image_shape))
        return avg_recon_err
    
    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history
    
    def evaluate(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        total_loss, recon_loss, kl_loss, re_kl_loss = \
            self.variational_ae.evaluate(X_inputs + Z_inputs, verbose=0)
        return total_loss, recon_loss, kl_loss, re_kl_loss
    
    def evaluate_gen(self, generator):
        steps = divide(generator.n, generator.batch_size)
        gen = custom_generator_fit(generator, self.epochs, with_RE=True)
        total_loss, recon_loss, kl_loss, re_kl_loss = \
            self.variational_ae.evaluate(gen, steps=steps, verbose=0)
        return total_loss, recon_loss, kl_loss, re_kl_loss


class LMMVAEIMGCNN:
    """LMMVAE Class for images with convolutional encoders/decoders

    """

    def __init__(self, mode, img_height, img_width, channels, qs, q_spatial, d, n_sig2bs,
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
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.input_img_dim = img_height * img_width * channels
        self.input_dim_conv = (img_height, img_width, channels)
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
        X_input = Input(shape=(img_height, img_width, channels))
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
        z1 = add_layers_functional_conv2d(X_input, n_neurons, dropout, activation, self.input_dim_conv)
        # codings_mean = Dense(d, kernel_regularizer=Orthogonal(d))(z1)
        codings_mean = Dense(d)(z1)
        codings_log_var = Dense(d)(z1)
        z2 = add_layers_functional_conv2d(X_input, n_neurons_re, dropout, activation, self.input_dim_conv)
        re_codings_mean_list = []
        re_codings_log_var_list = []
        re_codings_list = []
        for _ in range(self.n_RE_outputs):
            re_codings_mean = Dense(self.input_img_dim)(z2)
            re_codings_mean_list.append(re_codings_mean)
            re_codings_log_var = Dense(self.input_img_dim)(z2)
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
            decoder_re_inputs = Input(shape=self.input_img_dim)
            decoder_re_inputs_list.append(decoder_re_inputs)
        
        n_neurons_rev = None if n_neurons is None else list(reversed(n_neurons))
        dropout_rev = None if dropout is None else list(reversed(dropout))
        decoder_output = add_layers_functional_conv2d_t(decoder_inputs, n_neurons_rev, dropout_rev, activation, d, img_height, img_width, channels)
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
            ZB = Reshape((img_height, img_width, channels))(ZB)
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
        recon_loss = MeanSquaredError()(X_input, reconstructions) * self.input_img_dim
        self.variational_ae.add_loss(recon_loss)
        self.variational_ae.add_metric(recon_loss, name='recon_loss')
        self.variational_ae.add_metric(kl_loss, name='kl_loss')
        self.variational_ae.add_metric(re_kl_loss, name='re_kl_loss')
        self.variational_ae.compile(optimizer='adam')

    def _fit(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        if self.pred_unknown_clusters:
            Z_inputs_train, Z_inputs_valid = train_test_split(Z_inputs[0].unique(), test_size=0.1)
            Z_inputs_train = [Z_input[Z_inputs[0].isin(Z_inputs_train)] for Z_input in Z_inputs]
            Z_inputs_valid =  [Z_input[Z_inputs[0].isin(Z_inputs_valid)] for Z_input in Z_inputs]
            X_input_train = X_inputs[0].loc[Z_inputs_train[0].index]
            X_input_valid = X_inputs[0].loc[Z_inputs_valid[0].index]
            X_train = X.loc[Z_inputs_train[0].index]
            X_valid = X.loc[Z_inputs_valid[0].index]
            self.history = self.variational_ae.fit([X_input_train] + Z_inputs_train, X_train, epochs=self.epochs,
                callbacks=self.callbacks, batch_size=self.batch_size, validation_data=([X_input_valid] + Z_inputs_valid, X_valid),
                verbose=self.verbose)
        else:
            self.history = self.variational_ae.fit(X_inputs + Z_inputs, X, epochs=self.epochs,
                callbacks=self.callbacks, batch_size=self.batch_size, validation_split=0.1,
                verbose=self.verbose)
        gc.collect()

    def _fit_gen(self, train_generator, valid_generator):
        steps_train = divide(train_generator.n, train_generator.batch_size)
        steps_valid = divide(valid_generator.n, valid_generator.batch_size)
        train_gen = custom_generator_fit(train_generator, self.epochs, with_RE=True)
        valid_gen = custom_generator_fit(valid_generator, self.epochs, with_RE=True)
        self.history = self.variational_ae.fit(train_gen,
                                               validation_data=valid_gen,
                                               epochs=self.epochs, callbacks=self.callbacks,
                                               batch_size=self.batch_size, verbose=self.verbose,
                                               steps_per_epoch = steps_train,
                                               validation_steps = steps_valid)
        gc.collect()
    
    def fit(self, X, Z):
        self._fit(X, Z)
        return self

    def _get_input(self, X, Z):
        X_inputs = [X]
        Z_inputs = [Z[:, RE_col] for RE_col in range(len(self.qs))]
        return X_inputs, Z_inputs
    
    def _transform(self, X, Z, U, B_list, extract_B):
        X_inputs, Z_inputs = self._get_input(X, Z)
        encoder_output = self.variational_encoder.predict(X_inputs, verbose=0)
        X_transformed = encoder_output[0]
        B_hat_list = encoder_output[1:]
        if extract_B:
            B_hat_list_processed = self.extract_Bs_to_compare(Z_inputs, B_hat_list)
            sig2bs_hat_list = [B_hat_list_processed[i].var(axis=0) for i in range(len(B_hat_list_processed))]
            return X_transformed, B_hat_list_processed, sig2bs_hat_list
        else:
            return X_transformed, None, None
    
    def _transform_gen(self, generator, U, B_list, extract_B):
        steps = divide(generator.n, generator.batch_size)
        generator.reset()
        prev_shuffle_state = generator.shuffle
        generator.shuffle = False
        gen = custom_generator_predict(generator, self.epochs, with_RE=False)
        encoder_output = self.variational_encoder.predict(gen, steps=steps, verbose=0)
        X_transformed = encoder_output[0]
        B_hat_list = encoder_output[1:]
        if extract_B:
            Z_inputs = get_full_RE_cols_from_generator(generator)
            B_hat_list_processed = self.extract_Bs_to_compare(Z_inputs, B_hat_list)
            sig2bs_hat_list = [B_hat_list_processed[i].var(axis=0) for i in range(len(B_hat_list_processed))]
            generator.shuffle = prev_shuffle_state
            return X_transformed, B_hat_list_processed, sig2bs_hat_list
        else:
            generator.shuffle = prev_shuffle_state
            return X_transformed, None, None
    
    def extract_Bs_to_compare(self, Z_inputs, B_hat_list):
        B_df2_list = []
        for i in range(self.n_RE_outputs):
            B_df = pd.DataFrame(B_hat_list[i])
            q_ind = 0 if self.mode == 'longitudinal' else i
            B_df['z'] = Z_inputs[q_ind]
            B_df2 = B_df.groupby('z')[B_df.columns[:self.input_img_dim]].mean()
            B_df2 = B_df2.reindex(range(self.qs[q_ind]), fill_value=0)
            if self.mode == 'spatial' or (self.mode == 'spatial_and_categorical' and i == 0):
                B_df2 = pd.DataFrame(self.kernel_root.numpy() @ B_df2.values)
            # convert to image dims
            B_df2 = B_df2.values.reshape(B_df2.shape[0], self.img_height, self.img_width, self.channels)
            B_df2_list.append(B_df2)
        return B_df2_list

    def transform(self, X, Z, U, B_list, extract_B=False):
        check_is_fitted(self, 'history')
        return self._transform(X, Z, U, B_list, extract_B)

    def transform_gen(self, generator, U, B_list, extract_B=False):
        check_is_fitted(self, 'history')
        return self._transform_gen(generator, U, B_list, extract_B)
    
    def fit_transform(self, X, Z, U, B_list, reconstruct_B=True):
        self._fit(X, Z)
        return self._transform(X, Z, U, B_list, reconstruct_B)

    def fit_transform_gen(self, train_generator, valid_generator, U, B_list, reconstruct_B=True):
        self._fit_gen(train_generator, valid_generator)
        return self._transform_gen(train_generator, U, B_list, reconstruct_B)

    def reconstruct(self, X_transformed, Z_idxs, B_list):
        X_reconstructed = self.variational_decoder_no_re.predict([X_transformed], verbose=0)
        if self.mode == 'longitudinal':
            Z0 = sparse.csr_matrix(get_dummies(Z_idxs[:, 0], self.qs[0]))
            t = Z_idxs[:, 1]
            n = X_transformed.shape[0]
            for k in range(self.n_sig2bs):
                Z = sparse.spdiags(t ** k, 0, n, n) @ Z0
                X_reconstructed += Z @ B_list[k]
        else:
            for i in range(Z_idxs.shape[1]):
                Z = get_dummies(Z_idxs[:, i], self.qs[i])
                X_reconstructed += np.tensordot(Z.toarray(),  B_list[i], axes=[1,0])
        return X_reconstructed
    
    def recon_error_on_batches(self, generator, X_transformed, B_list):
        generator.reset()
        steps = divide(generator.n, generator.batch_size)
        total_recon_err = 0
        for i in range(steps):
            batch, Z_idxs = generator.next()
            idx = np.arange(i * generator.batch_size, i * generator.batch_size + batch.shape[0])
            batch_reconstructed = self.variational_decoder_no_re.predict_on_batch([X_transformed[idx]])
            if len(Z_idxs.shape) == 1:
                Z_idxs = Z_idxs[:, np.newaxis]
            for i in range(Z_idxs.shape[1]):
                Z = get_dummies(Z_idxs[:, i], self.qs[i])
                batch_reconstructed += np.tensordot(Z.toarray(),  B_list[i], axes=[1,0])
            total_recon_err += np.sum((batch - batch_reconstructed)**2)
        avg_recon_err = total_recon_err / (generator.n * np.prod(generator.image_shape))
        return avg_recon_err
    
    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history
    
    def evaluate(self, X, Z):
        X_inputs, Z_inputs = self._get_input(X, Z)
        total_loss, recon_loss, kl_loss, re_kl_loss = \
            self.variational_ae.evaluate(X_inputs + Z_inputs, verbose=0)
        return total_loss, recon_loss, kl_loss, re_kl_loss
    
    def evaluate_gen(self, generator):
        steps = divide(generator.n, generator.batch_size)
        gen = custom_generator_fit(generator, self.epochs, with_RE=True)
        total_loss, recon_loss, kl_loss, re_kl_loss = \
            self.variational_ae.evaluate(gen, steps=steps, verbose=0)
        return total_loss, recon_loss, kl_loss, re_kl_loss
