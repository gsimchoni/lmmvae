import os
import gc

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from svgpvae.utils import *
from svgpvae.classes import *
from svgpvae.actions import *


class SVGPVAE:
    """SVGPVAE Class

    """

    def __init__(self, d, q, x_cols, batch_size, epochs, patience, n_neurons, dropout, activation, verbose,
            M, nr_inducing_units, nr_inducing_per_unit, RE_cols, aux_cols, beta, GECO, disable_gpu) -> None:
        
        self.L = d
        self.q = q
        self.x_cols = x_cols
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.activation = activation
        self.verbose = verbose
        self.M = M
        self.nr_inducing_units = nr_inducing_units
        self.nr_inducing_per_unit = nr_inducing_per_unit
        self.RE_cols = RE_cols
        self.aux_cols = aux_cols
        self.beta_arg = beta
        self.GECO = GECO
        self.n_epochs = 0
        
        if disable_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        

    def run(self, X_train, X_test, valid_split=0.1):
        elbo_arg = 'SVGPVAE_Hensman'
        init_PCA=True
        ip_joint=True
        GP_joint=True
        ov_joint=True
        lr_arg=0.001
        alpha_arg=0.99
        jitter=0.000001
        object_kernel_normalize=False
        kappa_squared=0.020
        clip_qs=True
        opt_regime=['joint']
        ram=1.0

        # split train to train and valid
        X_train_new, X_valid = train_test_split(X_train, test_size=valid_split)

        train_data_dict, valid_data_dict, test_data_dict = process_data_for_svgpvae(
            X_train_new, X_test, X_valid, self.x_cols, self.aux_cols, self.RE_cols, self.M)
        
        graph = tf.Graph()
        with graph.as_default():
            train_data, _ = tensor_slice(train_data_dict, self.batch_size, placeholder=False)
            N_train = train_data_dict['data_Y'].shape[0]
            N_valid = valid_data_dict['data_Y'].shape[0]
            N_test = test_data_dict['data_Y'].shape[0]

            # valid data
            valid_data, valid_batch_size_placeholder = tensor_slice(valid_data_dict, self.batch_size, placeholder=True)

            # test data
            test_data, test_batch_size_placeholder = tensor_slice(test_data_dict, self.batch_size, placeholder=True)

            # init iterator
            iterator = tf.compat.v1.data.Iterator.from_structure(
                tf.compat.v1.data.get_output_types(train_data),
                tf.compat.v1.data.get_output_shapes(train_data)
            )
            training_init_op = iterator.make_initializer(train_data)
            valid_init_op = iterator.make_initializer(valid_data)
            test_init_op = iterator.make_initializer(test_data)

            # get the batch
            input_batch = iterator.get_next()

            # ====================== 2) build ELBO graph ======================

            # init VAE object
            VAE = TabularVAE(input_batch[0].shape[1], self.L, self.n_neurons, self.dropout, self.activation)
            beta = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

            # placeholders
            y_shape = (None,) + train_data_dict['data_Y'].shape[1:]
            train_aux_X_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, len(self.RE_cols) + len(self.aux_cols) + self.M))
            train_data_Y_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=y_shape)
            test_aux_X_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, len(self.RE_cols) + len(self.aux_cols) + self.M))
            test_data_Y_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=y_shape)

            inducing_points_init = generate_init_inducing_points_tabular(
                train_data_dict, self.RE_cols, self.aux_cols, self.nr_inducing_per_unit,
                self.nr_inducing_units, init_PCA, self.M)
            
            titsias = 'Titsias' in elbo_arg
            ip_joint = not ip_joint
            GP_joint = not GP_joint
            if ov_joint:
                if init_PCA:  # use PCA embeddings for initialization of object vectors
                    PC_cols = train_data_dict['aux_X'].columns[train_data_dict['aux_X'].columns.str.startswith('PC')]
                    object_vectors_init = train_data_dict['aux_X'].groupby('z0')[PC_cols].mean()
                    object_vectors_init = object_vectors_init.reindex(range(self.q), fill_value=0)
                else:  # initialize object vectors randomly
                    object_vectors_init = np.random.normal(0, 1.5, self.q * self.M).reshape(self.q, self.M)
            else:
                object_vectors_init = None

            # init SVGP object
            SVGP_ = TabularDataSVGP(titsias=titsias, fixed_inducing_points=ip_joint,
                            initial_inducing_points=inducing_points_init,
                            fixed_gp_params=GP_joint, object_vectors_init=object_vectors_init, name='main',
                            jitter=jitter, N_train=N_train,
                            L=self.L, K_obj_normalize=object_kernel_normalize,
                            RE_cols=self.RE_cols, aux_cols=self.aux_cols)

            # forward pass SVGPVAE
            C_ma_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
            lagrange_mult_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
            alpha_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

            elbo, recon_loss, KL_term, inside_elbo, ce_term, p_m, p_v, qnet_mu, qnet_var, recon_data_Y, \
            inside_elbo_recon, inside_elbo_kl, latent_samples, \
            C_ma, lagrange_mult, mean_vectors = forward_pass_SVGPVAE_tabular(input_batch,
                                                                    beta=beta,
                                                                    vae=VAE,
                                                                    svgp=SVGP_,
                                                                    C_ma=C_ma_placeholder,
                                                                    lagrange_mult=lagrange_mult_placeholder,
                                                                    alpha=alpha_placeholder,
                                                                    kappa=np.sqrt(kappa_squared),
                                                                    clipping_qs=clip_qs,
                                                                    GECO=self.GECO,
                                                                    bias_analysis=False)

            # forward pass standard VAE (for training regime from CASALE: VAE-GP-joint)
            recon_loss_VAE, KL_term_VAE, elbo_VAE, \
            recon_data_Y_VAE, qnet_mu_VAE, qnet_var_VAE, \
            latent_samples_VAE = forward_pass_standard_VAE(input_batch, vae=VAE)
            
            train_encodings_means_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, self.L))
            train_encodings_vars_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, self.L))

            qnet_mu_train, qnet_var_train, _ = batching_encode_SVGPVAE(input_batch, vae=VAE,
                                                                    clipping_qs=clip_qs)
            recon_data_Y_test, \
            recon_loss_test = batching_predict_SVGPVAE(input_batch,
                                                    vae=VAE,
                                                    svgp=SVGP_,
                                                    qnet_mu=train_encodings_means_placeholder,
                                                    qnet_var=train_encodings_vars_placeholder,
                                                    aux_data_train=train_aux_X_placeholder)

            # GP diagnostics
            GP_l, GP_amp, GP_ov, GP_ip = SVGP_.variable_summary()

            # ====================== 3) optimizer ops ======================
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            lr = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

            if self.GECO:  # minimizing GECO objective
                gradients = tf.gradients(elbo, train_vars)
            else:  # minimizing negative elbo
                gradients = tf.gradients(-elbo, train_vars)

            optim_step = optimizer.apply_gradients(grads_and_vars=zip(gradients, train_vars),
                                                global_step=global_step)

            init_op = tf.compat.v1.global_variables_initializer()

            # ====================== 6) saver and GPU ======================
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=ram)

            # ====================== 7) tf.session ======================
            opt_regime = [r + '-' + str(self.epochs) for r in opt_regime]
            nr_epochs, training_regime = parse_opt_regime(opt_regime)

            with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:

                sess.run(init_op)

                # training loop
                first_step = True  # switch for initialization of GECO algorithm
                C_ma_ = 0.0
                lagrange_mult_ = 1.0

                cgen_test_set_MSE = []

                best_valid_loss = np.inf
                best_loss_counter = 0
                stop_training = False

                def generator():
                    while True:
                        yield

                for epoch in range(nr_epochs):
                    # 7.1) train for one epoch
                    sess.run(training_init_op)
                    elbos, losses = [], []
                    batch = 0
                    t = tqdm(generator(), total=int(N_train / self.batch_size), ascii=True, disable=not self.verbose)
                    for _ in t:
                        t.set_description(f'epoch {epoch+1}/{nr_epochs}:')
                        try:
                            if self.GECO and training_regime[epoch] != 'VAE':
                                if first_step:
                                    alpha = 0.0
                                else:
                                    alpha = alpha_arg
                                _, g_s_, elbo_, C_ma_, lagrange_mult_, recon_loss_, mean_vectors_ = sess.run([optim_step, global_step,
                                                                                elbo, C_ma, lagrange_mult,
                                                                                recon_loss, mean_vectors],
                                                                                {beta: self.beta_arg, lr: lr_arg,
                                                                                alpha_placeholder: alpha,
                                                                                C_ma_placeholder: C_ma_,
                                                                                lagrange_mult_placeholder: lagrange_mult_})
                            else:
                                _, g_s_, elbo_, recon_loss_ = sess.run([optim_step, global_step, elbo, recon_loss],
                                                        {beta: self.beta_arg, lr: lr_arg,
                                                        alpha_placeholder: alpha_arg,
                                                        C_ma_placeholder: C_ma_,
                                                        lagrange_mult_placeholder: lagrange_mult_})
                            elbos.append(elbo_)
                            losses.append(recon_loss_)
                            first_step = False  # switch for initizalition of GECO algorithm
                            batch += 1
                            t.set_postfix({'train_loss': round(recon_loss_/self.batch_size, 4)})
                        except tf.errors.OutOfRangeError:
                            break

                    # 7.2) calculate loss on valid set
                    losses = []
                    sess.run(valid_init_op, {valid_batch_size_placeholder: self.batch_size})
                    while True:
                        try:
                            recon_loss_ = sess.run(recon_loss, {beta: self.beta_arg, lr: lr_arg,
                                                                alpha_placeholder: alpha_arg,
                                                                C_ma_placeholder: C_ma_,
                                                                lagrange_mult_placeholder: lagrange_mult_})
                            losses.append(recon_loss_)
                        except tf.errors.OutOfRangeError:
                            MSE_valid = np.sum(losses) / N_valid
                            if self.verbose:
                                t.write(f'valid_loss: {round(MSE_valid, 4)}')
                            # early stopping
                            if MSE_valid < best_valid_loss:
                                best_valid_loss = MSE_valid
                                best_loss_counter = 0
                            else:
                                best_loss_counter += 1
                                if best_loss_counter == self.patience:
                                    stop_training = True
                            break
                    if stop_training:
                        break

                # 7.4) calculate loss on test set and visualize reconstructed data
                losses, recon_data_Y_arr = [], []
                sess.run(test_init_op, {test_batch_size_placeholder: self.batch_size})
                # test set: reconstruction
                while True:
                    try:
                        recon_loss_, recon_data_Y_ = sess.run([recon_loss, recon_data_Y],
                                                                {beta: self.beta_arg,
                                                                alpha_placeholder: alpha_arg,
                                                                C_ma_placeholder: C_ma_,
                                                                lagrange_mult_placeholder: lagrange_mult_})
                        losses.append(recon_loss_)
                        recon_data_Y_arr.append(recon_data_Y_)
                    except tf.errors.OutOfRangeError:
                        MSE = np.sum(losses) / N_test
                        if self.verbose:
                            print('MSE loss on test set for epoch {} : {}'.format(epoch, MSE))
                        recon_data_Y_arr = np.concatenate(tuple(recon_data_Y_arr))
                        # plot_tabular(test_data_dict['data_Y'],
                        #             recon_data_Y_arr,
                        #             title="Epoch: {}. Recon MSE test set:{}".format(epoch + 1, round(MSE, 4)))
                        break
                
                # test set: conditional generation SVGPVAE
                # encode training data (in batches)
                sess.run(training_init_op)
                means, vars = [], []
                while True:
                    try:
                        qnet_mu_train_, qnet_var_train_ = sess.run([qnet_mu_train, qnet_var_train])
                        means.append(qnet_mu_train_)
                        vars.append(qnet_var_train_)
                    except tf.errors.OutOfRangeError:
                        break
                means = np.concatenate(means, axis=0)
                vars = np.concatenate(vars, axis=0)

                # predict test data (in batches)
                sess.run(test_init_op, {test_batch_size_placeholder: self.batch_size})
                recon_loss_cgen, recon_data_Y_cgen = [], []
                while True:
                    try:
                        loss_, recon_Y_batch_ = sess.run([recon_loss_test, recon_data_Y_test],
                                                {train_aux_X_placeholder: train_data_dict['aux_X'],
                                                    train_encodings_means_placeholder: means,
                                                    train_encodings_vars_placeholder: vars})
                        recon_loss_cgen.append(loss_)
                        recon_data_Y_cgen.append(recon_Y_batch_)
                    except tf.errors.OutOfRangeError:
                        break
                recon_loss_cgen = np.sum(recon_loss_cgen) / N_test
                recon_data_Y_cgen = np.concatenate(recon_data_Y_cgen, axis=0)

                cgen_test_set_MSE.append((epoch, recon_loss_cgen))
                if self.verbose:
                    print("Conditional generation MSE loss on test set for epoch {}: {}".format(epoch, recon_loss_cgen))

        self.n_epochs = epoch
        return None, None, recon_data_Y_cgen

    def get_n_epochs(self):
        return self.n_epochs
