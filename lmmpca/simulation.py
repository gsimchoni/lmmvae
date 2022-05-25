import logging
from itertools import product

import pandas as pd

from lmmpca.pca import reg_pca
from lmmpca.utils import PCAInput, generate_data

logger = logging.getLogger('LMMPCA.logger')


class Count:
    curr = 0

    def __init__(self, startWith=None):
        if startWith is not None:
            Count.curr = startWith - 1

    def gen(self):
        while True:
            Count.curr += 1
            yield Count.curr


def iterate_reg_types(counter, res_df, out_file, pca_in, exp_types, verbose):
    if 'ignore' in exp_types:
        if verbose:
            logger.info('mode ignore:')
        res = run_reg_pca(pca_in, 'ignore')
        ig_res = summarize_sim(pca_in, res, 'ignore')
        res_df.loc[next(counter)] = ig_res
        logger.debug('  Finished Ignore.')
    res_df.to_csv(out_file)


def run_reg_pca(pca_in, reg_type):
    return reg_pca(pca_in.X_train, pca_in.X_test, pca_in.y_train, pca_in.y_test)


def summarize_sim(pca_in, res, reg_type):
    res = [pca_in.mode, pca_in.N, pca_in.sig2e] + list(pca_in.sig2bs) +\
        [res.n_epochs, res.time]
    return res


def simulation(out_file, params):
    counter = Count().gen()
    n_sig2bs = len(params['sig2b_list'])

    sig2bs_names = list(map(lambda x: 'sig2b' + str(x), range(n_sig2bs)))
    res_df = pd.DataFrame(
        columns=['mode', 'N', 'sig2e'] + sig2bs_names + ['n_epochs', 'time'])
    for N in params['N_list']:
        for sig2e in params['sig2e_list']:
            for qs in product(*params['q_list']):
                for sig2bs in product(*params['sig2b_list']):
                    logger.info('N: %d, sig2e: %.2f; sig2bs: [%s]' %
                                (N, sig2e, ', '.join(map(str, sig2bs))))
                    for k in range(params['n_iter']):
                        X_train, X_test, y_train, y_test, x_cols, dist_matrix, time2measure_dict = generate_data(
                            N, 10, qs, 2, sig2e, 1, sig2bs_identical=False)
                        logger.info(' iteration: %d' % k)
                        pca_in = PCAInput(
                            X_train, X_test, y_train, y_test, x_cols, N, qs, sig2e)
                        iterate_reg_types(
                            counter, res_df, out_file, pca_in, params['exp_types'], params['verbose'])
