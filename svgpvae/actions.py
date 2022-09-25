import tensorflow as tf
from svgpvae.utils import gauss_cross_entropy

def forward_pass_SVGPVAE_tabular(data_batch, beta, vae, svgp, C_ma, lagrange_mult, alpha,
                         kappa, clipping_qs=False, GECO=False,
                         repr_NN=None, segment_ids=None, repeats=None, bias_analysis=False):
    """
    Forward pass for SVGPVAE on tabular data.

    :param data_batch: (data_Y, aux_X). data_Y dimension: (batch_size, p + len(RE_cols) + len(aux_cols)).
        aux_X dimension: (batch_size, 10)
    :param beta:
    :param vae: VAE object
    :param svgp: SVGP object
    :param C_ma: average constraint from t-1 step (GECO)
    :param lagrange_mult: lambda from t-1 step (GECO)
    :param kappa: reconstruction level parameter for GECO
    :param alpha: moving average parameter for GECO
    :param clipping_qs: clipping of VAE posterior distribution (for numerical stability)
    :param GECO: whether or not to use GECO algorithm for training
    :param repr_NN: representation network (used only in case of SPRITES data)
    :param segment_ids: Used only in case of SPRITES data.
    :param repeats: Used only in case of SPRITES data.
    :param bias_analysis:

    :return:
    """

    data_Y, aux_X = data_batch
    y_shape = data_Y.get_shape()
    K = tf.cast(tf.reduce_prod(y_shape[1:]), dtype=vae.dtype)
    b = tf.cast(tf.shape(data_Y)[0], dtype=vae.dtype)  # batch_size

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(data_Y)
    L = tf.cast(qnet_mu.get_shape()[1], dtype=vae.dtype)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    # SVGP: inside-ELBO term (L_2 or L_3), approx posterior distribution
    inside_elbo_recon, inside_elbo_kl = [], []
    p_m, p_v = [], []
    for l in range(qnet_mu.get_shape()[1]):  # iterate over latent dimensions
        p_m_l, p_v_l, mu_hat_l, A_hat_l = svgp.approximate_posterior_params(aux_X, aux_X,
                                                                            qnet_mu[:, l], qnet_var[:, l])
        inside_elbo_recon_l,  inside_elbo_kl_l = svgp.variational_loss(x=aux_X, y=qnet_mu[:, l],
                                                                       noise=qnet_var[:, l], mu_hat=mu_hat_l,
                                                                       A_hat=A_hat_l)

        inside_elbo_recon.append(inside_elbo_recon_l)
        inside_elbo_kl.append(inside_elbo_kl_l)
        p_m.append(p_m_l)
        p_v.append(p_v_l)

    inside_elbo_recon = tf.reduce_sum(inside_elbo_recon)
    inside_elbo_kl = tf.reduce_sum(inside_elbo_kl)

    if svgp.titsias:
        inside_elbo = inside_elbo_recon - inside_elbo_kl
    else:
        inside_elbo = inside_elbo_recon - (b / svgp.N_train) * inside_elbo_kl

    p_m = tf.stack(p_m, axis=1)
    p_v = tf.stack(p_v, axis=1)

    if repr_NN:  # for numerical stability in SPRITES experiment
        p_v = tf.clip_by_value(p_v, 1e-4, 100)

    # cross entropy term
    ce_term = gauss_cross_entropy(p_m, p_v, qnet_mu, qnet_var)
    ce_term = tf.reduce_sum(ce_term)

    KL_term = -ce_term + inside_elbo

    # SAMPLE
    epsilon = tf.random.normal(shape=tf.shape(p_m), dtype=vae.dtype)
    latent_samples = p_m + epsilon * tf.sqrt(p_v)

    # DECODER NETWORK
    recon_data_Y_logits = tf.cast(vae.decode(latent_samples), tf.float64)
    recon_data_Y = tf.cast(recon_data_Y_logits, tf.float64)

    if GECO:
        recon_loss = tf.reduce_mean((data_Y - recon_data_Y_logits) ** 2, axis=1)
        recon_loss = tf.reduce_sum(recon_loss - kappa**2)
        C_ma = alpha * C_ma + (1 - alpha) * recon_loss / b

        elbo = - KL_term + lagrange_mult * (recon_loss/b + tf.stop_gradient(C_ma - recon_loss/b))

        lagrange_mult = lagrange_mult * tf.exp(C_ma)

    else:
        recon_loss = tf.reduce_sum((data_Y - recon_data_Y_logits) ** 2)

        # ELBO
        # beta plays role of sigma_gaussian_decoder here (\lambda(\sigma_y) in Casale paper)
        # K and L are not part of ELBO. They are used in loss objective to account for the fact that magnitudes of
        # reconstruction and KL terms depend on number of pixels (K) and number of latent GPs used (L), respectively
        recon_loss = recon_loss / K
        elbo = - recon_loss + (beta / L) * KL_term

    # bias analysis
    if bias_analysis:
        mean_vectors = []
        for l in range(qnet_mu.get_shape()[1]):
            mean_vectors.append(svgp.mean_vector_bias_analysis(aux_X, qnet_mu[:, l], qnet_var[:, l]))
    else:
        mean_vectors = tf.constant(1.0)  # dummy placeholder

    return elbo, recon_loss, KL_term, inside_elbo, ce_term, p_m, p_v, qnet_mu, qnet_var, \
           recon_data_Y, inside_elbo_recon, inside_elbo_kl, latent_samples, C_ma, lagrange_mult, mean_vectors


def KL_term_standard_normal_prior(mean_vector, var_vector, dtype):
    """
    Computes KL divergence between standard normal prior and variational distribution from encoder.

    :param mean_vector: (batch_size, L)
    :param var_vector:  (batch_size, L)
    :return: (batch_size, 1)
    """
    return 0.5 * (- tf.cast(tf.reduce_prod(tf.shape(mean_vector)), dtype=dtype)
                  - 2.0*tf.reduce_sum(tf.math.log(tf.sqrt(var_vector)))
                  + tf.reduce_sum(var_vector)
                  + tf.reduce_sum(mean_vector**2))


def forward_pass_standard_VAE(data_batch, vae, sigma_gaussian_decoder=0.01, clipping_qs=False):
    """
    Forward pass for SVGPVAE on data. This is plain VAE forward pass (used in VAE-GP-joint
    training regime).

    :param data_batch:
    :param vae:
    :param sigma_gaussian_decoder: standard deviation of Gaussian decoder

    :return:
    """

    data_Y, aux_X = data_batch
    # _, w, h, c = data_Y.get_shape()
    y_shape = data_Y.get_shape()  # for MNIST c==1, for SPRITES c==3
    b = tf.shape(data_Y)[0]

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(data_Y)
    qnet_mu = tf.cast(qnet_mu, tf.float64)
    qnet_var = tf.cast(qnet_var, tf.float64)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    # SAMPLE
    epsilon = tf.random.normal(shape=tf.shape(qnet_mu), dtype=vae.dtype)
    latent_samples = qnet_mu + epsilon * tf.sqrt(qnet_var)

    # DECODER NETWORK
    # could consider CE loss as well here (then would have Bernoulli decoder), but for that would then need to adjust
    # range of beta param. Note that sigmoid only makes sense for Bernoulli decoder
    recon_data_Y_logits = tf.cast(vae.decode(latent_samples), tf.float64)

    # Gaussian observational likelihood
    recon_data_Y = recon_data_Y_logits
    recon_loss = tf.reduce_sum((data_Y - recon_data_Y_logits) ** 2)

    # Bernoulli observational likelihood, CE
    # recon_images = tf.nn.sigmoid(recon_images_logits)
    # recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=images,
    #                                                                    logits=recon_images_logits))

    # ELBO (plain VAE)
    KL_term = KL_term_standard_normal_prior(qnet_mu, qnet_var, dtype=vae.dtype)

    elbo = -(0.5/sigma_gaussian_decoder**2)*recon_loss - KL_term

    # report MSE per pixel
    K = tf.cast(tf.reduce_prod(y_shape[1:]), dtype=vae.dtype)
    recon_loss = recon_loss / K

    return recon_loss, KL_term, elbo, recon_data_Y, qnet_mu, qnet_var, latent_samples


def batching_encode_SVGPVAE(data_batch, vae, clipping_qs=False, repr_nn=None,
                            segment_ids=None, repeats=None):
    """
    This function encodes images to latent representations in batches for SVGPVAE model.
    
    :param data_batch:
    :param vae:
    :param clipping_qs:
    :param repr_nn: representation network. used only in case of SPRITES data
    :param segment_ids: used only in case of SPRITES data
    :param repeats: used only in case of SPRITES data
    :return: 
    """

    images, aux_data = data_batch

    b = tf.shape(images)[0]

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(images)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    return qnet_mu, qnet_var, aux_data


def batching_predict_SVGPVAE(test_data_batch, vae, svgp,
                             qnet_mu, qnet_var, aux_data_train):
    """
    Get predictions for test data. See chapter 3.3 in Casale's paper.
    This version supports batching in prediction pipeline (contrary to function predict_SVGPVAE_rotated_mnist) .

    :param test_data_batch: batch of test data
    :param vae: fitted (!) VAE object
    :param svgp: fitted (!) SVGP object
    :param qnet_mu: precomputed encodings (means) of train dataset (N_train, L)
    :param qnet_var: precomputed encodings (vars) of train dataset (N_train, L)
    :param aux_data_train: train aux data (N_train, 10)
    :return:
    """
    data_Y_test_batch, aux_X_test_batch = test_data_batch

    y_shape = data_Y_test_batch.get_shape()

    # get latent samples for test data from GP posterior
    p_m, p_v = [], []
    for l in range(qnet_mu.get_shape()[1]):  # iterate over latent dimensions
        p_m_l, p_v_l, _, _ = svgp.approximate_posterior_params(index_points_test=aux_X_test_batch,
                                                               index_points_train=aux_data_train,
                                                               y=qnet_mu[:, l], noise=qnet_var[:, l])
        p_m.append(p_m_l)
        p_v.append(p_v_l)

    p_m = tf.stack(p_m, axis=1)
    p_v = tf.stack(p_v, axis=1)

    epsilon = tf.random.normal(shape=tf.shape(p_m), dtype=tf.float64)
    latent_samples = p_m + epsilon * tf.sqrt(p_v)

    # predict (decode) latent images.
    # ===============================================
    # Since this is generation (testing pipeline), could add \sigma_y to images
    recon_data_Y_test_logits = tf.cast(vae.decode(latent_samples), tf.float64)

    # Gaussian observational likelihood, no variance
    recon_data_Y_test = recon_data_Y_test_logits

    # Bernoulli observational likelihood
    # recon_images_test = tf.nn.sigmoid(recon_images_test_logits)

    # Gaussian observational likelihood, fixed variance \sigma_y
    # recon_images_test = recon_images_test_logits + tf.random.normal(shape=tf.shape(recon_images_test_logits),
    #                                                                 mean=0.0, stddev=0.04, dtype=tf.float64)

    # MSE loss for CGEN (here we do not consider MSE loss, ince )
    recon_loss = tf.reduce_sum((data_Y_test_batch - recon_data_Y_test_logits) ** 2)

    # report per pixel loss
    K = tf.cast(tf.reduce_prod(y_shape[1:]), dtype=vae.dtype)
    recon_loss = recon_loss / K
    # ===============================================

    return recon_data_Y_test, recon_loss

