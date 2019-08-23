#!/usr/bin/env python

import pickle
import argparse
import ot.gpu
import ot
import numpy as np
from utils import instrument, chrono
import os
from scipy.spatial.distance import pdist, squareform

import logging

logger = logging.getLogger()

# TODO: add nonconvex group lasso reg as a technique:
# reg: $\eta \times \Omega_g(\gamma)$ where $\Omega_g(\gamma) = \sum_{i,c} \norm{\gamma_{i,I_c}_1^{1/2}$
# I_c: index of samples from class $c$ in the source domain.
@instrument
def compute_all_wasserstein_from(src,
        user_distributions,
        M,
        sinkhorn_reg,
        lasso_reg,
        enable_non_convex_lasso,
        max_iterations,
        verbose,
        use_cpu):
    try:
        if enable_non_convex_lasso:
            if max_iterations >= 1000:
                logger.warning('When you enable non-convex LASSO for Sinkhorn algorithm, the default max iterations (of the Sinkhorn solver) is around ~ 200, are you sure to proceed with {} max iterations?'.format(max_iterations))
            
            raise NotImplementedError('`a_labels` (I_c) are not well defined here yet.')
            
            sinkhorn = ot.gpu.sinkhorn_lpl1_mm if not use_cpu else ot.sinkhorn_lpl1_mm

            return sinkhorn(src,
                    None,
                    user_distributions.T,
                    M,
                    sinkhorn_reg,
                    lasso_reg,
                    numInnerItermax=max_iterations,
                    verbose=verbose,
                    to_numpy=True)
        else:
            sinkhorn = ot.gpu.sinkhorn if not use_cpu else ot.sinkhorn
            return sinkhorn(
                    src,
                    user_distributions.T,
                    M,
                    sinkhorn_reg,
                    numItermax=max_iterations,
                    verbose=verbose,
                    to_numpy=True)
    except Exception as e:
        logger.exception(e)

@instrument
def compute_all_wasserstein(user_distributions,
        M,
        sinkhorn_reg=1.0,
        lasso_reg=0.1,
        enable_non_convex_lasso=False,
        max_iterations=1000,
        verbose=False,
        use_cpu=False):
    ws = []
    for user_id in user_distributions:
        ws.append(compute_all_wasserstein_from(
                    user_id,
                    user_distributions,
                    M,
                    sinkhorn_reg,
                    lasso_reg,
                    enable_non_convex_lasso,
                    max_iterations,
                verbose,
                use_cpu)
            )

    return np.vstack(ws)

def load_ot_data_in_memory(input_ot_workload, use_cpu: bool = False):
    with open(input_ot_workload, 'rb') as f:
        _, C, user_distributions, _ = pickle.load(f)

        if use_cpu:
            return C, user_distributions
        else:
            return ot.gpu.to_gpu(C, user_distributions)

def compute_squareform(cost_matrix, metric, use_cpu: bool = False):
    if use_cpu:
        return squareform(pdist(cost_matrix, metric))
    else:
        return ot.gpu.dist(cost_matrix, metric=metric, to_numpy=False)

def main():
    parser = argparse.ArgumentParser(prog='gpu_sinkhorn', description='Run Sinkhorn on GPU over prepared OT data')
    parser.add_argument('input_ot_workload', help='The input NumPy arrays pickled file in the format specified in README.md')
    parser.add_argument('output_wasserstein_matrix', help='The output NumPy matrix which are the pairwise Wasserstein distances')
    parser.add_argument('--chrono', action='store_true', help='Enable the chronometer')
    parser.add_argument('--use-cpu', action='store_true', help='Disable GPU usage (useful for debugging as this is GPU/CPU agnostic)')
    parser.add_argument('--enable-pot-logging', action='store_true', help='Pass verbose=True to POT functions')
    parser.add_argument('--non-convex-lasso', action='store_true', help='Enable non-convex LASSO regularization (experimental)')
    parser.add_argument('--metric', default='sqeuclidean', help='Metric to pass to scipy.spatial.distance.pdist for pairwise computation (default: sqeuclidean)')
    parser.add_argument('-v', '--verbose', dest='verbose_count', action='count',
            default=0, help='Increases the log verbosity for each occurrence')
    parser.add_argument('-r', '--sinkhorn-regularization', default=1, type=float, help='Sinkhorn entropic regularization')
    parser.add_argument('-i', '--sinkhorn-max-iterations', default=1000, help='Max number of iterations for Sinkhorn')
    parser.add_argument('-e', '--lasso-regularization', default=0.1, type=float, help='Nonconvex LASSO regularization')
    args = parser.parse_args()

    effectiveLevel = max(3 - max(args.verbose_count, int(args.chrono)), 0) * 10
    logger.setLevel(effectiveLevel)

    logging.info('Logging initialized at level: {}'.format(effectiveLevel))

    chrono.is_enabled = args.chrono
    chrono.save('Argument parsing')

    if not os.path.isfile(args.input_ot_workload) and not os.access(args.input_ot_workload, os.R_OK):
        raise ValueError('{} is not reachable (permissions or does not exist)'.format(args.input_ot_workload))

    if os.path.isfile(args.output_wasserstein_matrix) and not os.access(args.output_wasserstein_matrix, os.W_OK):
        raise ValueError('{} is not writeable (permissions)'.format(args.output_wasserstein_matrix))

    C, user_distributions = load_ot_data_in_memory(args.input_ot_workload, args.use_cpu)
    chrono.save('OT data loaded in memory')
    logger.info('C[{}]: {}'.format(C.shape, C))
    logger.info('User distributions[{}]: {}'.format(user_distributions.shape, user_distributions))
    M = compute_squareform(C, args.metric, args.use_cpu)
    chrono.save('Pairwise matrix of distances ({}) computed'.format(M.shape))

    ws_matrix = compute_all_wasserstein(user_distributions, M, args.sinkhorn_regularization,
        args.lasso_regularization, args.non_convex_lasso,
        args.sinkhorn_max_iterations, args.enable_pot_logging,
        args.use_cpu)
    chrono.save('All Wasserstein were computed')

    with open(args.output_wasserstein_matrix, 'wb') as f:
        f.write(pickle.dumps(ws_matrix, pickle.HIGHEST_PROTOCOL))

    chrono.save('Wasserstein matrix wrote on disk')

if __name__ == '__main__':
    main()
