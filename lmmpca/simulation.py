import logging
import os
from itertools import product

import pandas as pd

from lmmpca.regression import reg_pca
from lmmpca.utils import PCAInput, generate_data

logger = logging.getLogger('LMMPCA.logger')
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


def iterate_pca_types(counter, res_df, out_file, pca_in, pca_types, verbose):
    for pca_type in pca_types:
        if verbose:
            logger.info(f'mode {pca_type}')
        res = run_reg_pca(pca_in, pca_type)
        res_summary = summarize_sim(pca_in, res, pca_type)
        res_df.loc[next(counter)] = res_summary
        logger.debug(f'  Finished {pca_type}.')
    res_df.to_csv(out_file)


def run_reg_pca(pca_in, pca_type):
    return reg_pca(pca_in.X_train, pca_in.X_test, pca_in.y_train,
                   pca_in.y_test, pca_in.x_cols, pca_in.RE_col, pca_in.d, pca_type,
                   pca_in.thresh, pca_in.epochs, pca_in.q, pca_in.batch_size,
                   pca_in.patience, pca_in.n_neurons, pca_in.dropout, pca_in.activation,
                   pca_in.verbose, pca_in.U, pca_in.B)


def summarize_sim(pca_in, res, pca_type):
    res = [pca_in.N, pca_in.p, pca_in.q, pca_in.d, pca_in.sig2e, pca_in.sig2bs_mean,
           pca_in.sig2bs_identical, pca_in.thresh, pca_in.k, pca_type, res.metric,
           res.sigmas[0], res.sigmas[1], res.n_epochs, res.time]
    return res


def simulation(out_file, params):
    counter = Count().gen()
    res_df = pd.DataFrame(
        columns=['N', 'p', 'q', 'd', 'sig2e', 'sig2bs_mean', 'sig2bs_identical', 'thresh'] +
        ['experiment', 'exp_type', 'mse', 'sig2e_est', 'sig2bs_mean_est', 'n_epochs', 'time'])
    for N in params['N_list']:
        for sig2e in params['sig2e_list']:
            for qs in product(*params['q_list']):
                for sig2bs_mean in params['sig2bs_mean_list']:
                    for sig2bs_identical in params['sig2bs_identical_list']:
                        for latend_dimension in params['latent_dimension_list']:
                            logger.info(f'N: {N}, q: {qs[0]}, d: {latend_dimension}, '
                                        f'sig2e: {sig2e:.2f}, sig2bs_mean: {sig2bs_mean}, '
                                        f'sig2bs_identical: {sig2bs_identical}')
                            for k in range(params['n_iter']):
                                pca_data = generate_data(N, qs, latend_dimension,
                                                         sig2e, sig2bs_mean, sig2bs_identical, params)
                                logger.info(' iteration: %d' % k)
                                pca_in = PCAInput(*pca_data, N, params['n_fixed_features'],
                                                  qs[0], latend_dimension,
                                                  sig2e, sig2bs_mean, sig2bs_identical, k,
                                                  params['epochs'], params['RE_col'],
                                                  params['thresh'], params['batch_size'],
                                                  params['patience'],
                                                  params['n_neurons'], params['dropout'],
                                                  params['activation'], params['verbose'])
                                iterate_pca_types(counter, res_df, out_file,
                                                  pca_in, params['pca_types'], params['verbose'])
