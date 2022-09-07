import logging
import os
from itertools import product

import pandas as pd

from lmmvae.regression import run_dim_reduction
from lmmvae.utils import DRInput, generate_data

logger = logging.getLogger('LMMVAE.logger')
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


class Count:
    curr = 0

    def __init__(self, startWith=None):
        if startWith is not None:
            Count.curr = startWith - 1

    def gen(self):
        while True:
            Count.curr += 1
            yield Count.curr


def iterate_dr_types(counter, res_df, out_file, dr_in, dr_types, verbose):
    for dr_type in dr_types:
        if verbose:
            logger.info(f'mode {dr_type}')
        res = run_dr(dr_in, dr_type)
        res_summary = summarize_sim(dr_in, res, dr_type)
        res_df.loc[next(counter)] = res_summary
        logger.debug(f'  Finished {dr_type}.')
    res_df.to_csv(out_file)


def run_dr(dr_in, dr_type):
    return run_dim_reduction(dr_in.X_train, dr_in.X_test, dr_in.x_cols,
                dr_in.RE_cols_prefix, dr_in.d, dr_type, dr_in.thresh,
                dr_in.epochs, dr_in.qs, dr_in.q_spatial, dr_in.n_sig2bs,
                dr_in.n_sig2bs_spatial, dr_in.estimated_cors, dr_in.batch_size,
                dr_in.patience, dr_in.n_neurons, dr_in.n_neurons_re, dr_in.dropout, dr_in.activation,
                dr_in.mode, dr_in.beta, dr_in.re_prior, dr_in.kernel, dr_in.verbose, dr_in.U, dr_in.B_list)


def summarize_sim(dr_in, res, dr_type):
    if dr_in.q_spatial is not None:
        q_spatial = [dr_in.q_spatial]
    else:
        q_spatial = []
    res = [dr_in.mode, dr_in.N, dr_in.p, dr_in.d, dr_in.sig2e, dr_in.beta, dr_in.re_prior] + \
        list(dr_in.qs) + q_spatial + list(dr_in.sig2bs_means) + list(dr_in.sig2bs_spatial) + list(dr_in.rhos) + \
        [dr_in.sig2bs_identical, dr_in.thresh, dr_in.k, dr_type,
        res.metric_X, res.sigmas[0]] + res.sigmas[1] + res.sigmas[2] + res.rhos + [res.n_epochs, res.time]
    return res


def simulation(out_file, params):
    mode = params['mode']
    n_sig2bs = len(params['sig2bs_mean_list'])
    n_sig2bs_spatial = len(params['sig2b_spatial_list'])
    n_categoricals = len(params['q_list'])
    sig2bs_spatial_names = []
    sig2bs_spatial_est_names = []
    q_spatial_name = []
    q_spatial_list = [None]
    n_rhos = len(params.get('rho_list', []))
    estimated_cors = params.get('estimated_cors', [])
    rhos_names =  []
    rhos_est_names =  []
    n_neurons_re = params.get('n_neurons_re', params['n_neurons'])
    if mode == 'categorical':
        assert n_sig2bs == n_categoricals
    elif mode in ['spatial', 'spatial_fit_categorical', 'spatial2']:
        assert n_categoricals == 0
        assert n_sig2bs == 0
        assert n_sig2bs_spatial == 2
        sig2bs_spatial_names = ['sig2b0_spatial', 'sig2b1_spatial']
        sig2bs_spatial_est_names = ['sig2b_spatial_est0', 'sig2b_spatial_est1']
        q_spatial_name = ['q_spatial']
        q_spatial_list = params['q_spatial_list']
    elif mode == 'longitudinal':
        assert n_categoricals == 1
        rhos_names =  list(map(lambda x: 'rho' + str(x), range(n_rhos)))
        rhos_est_names =  list(map(lambda x: 'rho_est' + str(x), range(len(estimated_cors))))
    else:
        raise ValueError('Unknown mode')
    qs_names =  list(map(lambda x: 'q' + str(x), range(n_categoricals)))
    sig2bs_names =  list(map(lambda x: 'sig2b' + str(x), range(n_sig2bs)))
    sig2bs_est_names =  list(map(lambda x: 'sig2b_est' + str(x), range(n_sig2bs)))
    beta_list = params.get('beta_list', [1/params['n_fixed_features']])
    re_prior = params.get('re_prior', 1.0)
    counter = Count().gen()
    res_df = pd.DataFrame(
        columns=['mode', 'N', 'p', 'd', 'sig2e', 'beta', 're_prior'] +
        qs_names + q_spatial_name + sig2bs_names +
        sig2bs_spatial_names + rhos_names + ['sig2bs_identical', 'thresh'] +
        ['experiment', 'exp_type', 'mse_X', 'sig2e_est'] + sig2bs_est_names +
        sig2bs_spatial_est_names + rhos_est_names + ['n_epochs', 'time'])
    for N in params['N_list']:
        for sig2e in params['sig2e_list']:
            for qs in product(*params['q_list']):
                for q_spatial in q_spatial_list:
                    for sig2bs_means in product(*params['sig2bs_mean_list']):
                        for sig2bs_spatial in product(*params['sig2b_spatial_list']):
                            for rhos in product(*params['rho_list']):
                                for sig2bs_identical in params['sig2bs_identical_list']:
                                    for latent_dimension in params['latent_dimension_list']:
                                        for beta in beta_list:
                                            logger.info(f'mode: {mode}, N: {N}, qs: {", ".join(map(str, qs))}, '
                                                        f'q_spatial: {q_spatial}, d: {latent_dimension}, '
                                                        f'sig2e: {sig2e}, sig2bs_mean: {", ".join(map(str, sig2bs_means))}, '
                                                        f'sig2bs_mean_spatial: {", ".join(map(str, sig2bs_spatial))}, '
                                                        f'rhos: {", ".join(map(str, rhos))}, '
                                                        f'sig2bs_identical: {sig2bs_identical}, beta: {beta}')
                                            for k in range(params['n_iter']):
                                                dr_data = generate_data(mode, N, qs, q_spatial, latent_dimension,
                                                                        sig2e, sig2bs_means, sig2bs_spatial, rhos, sig2bs_identical, params)
                                                logger.info(' iteration: %d' % k)
                                                dr_in = DRInput(*dr_data, mode, N, params['n_fixed_features'],
                                                                qs, latent_dimension,
                                                                sig2e, sig2bs_means, sig2bs_spatial, q_spatial, rhos, sig2bs_identical,
                                                                beta, re_prior, k, n_sig2bs,
                                                                n_sig2bs_spatial, estimated_cors,
                                                                params['epochs'], params['RE_cols_prefix'],
                                                                params['thresh'], params['batch_size'],
                                                                params['patience'],
                                                                params['n_neurons'], n_neurons_re, params['dropout'],
                                                                params['activation'], params['verbose'])
                                                iterate_dr_types(counter, res_df, out_file,
                                                                dr_in, params['dr_types'], params['verbose'])
