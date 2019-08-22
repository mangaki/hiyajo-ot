#!/usr/bin/env python

import pickle
import argparse
import ot
import numpy as np
from utils import instrument, chrono

# TODO: add nonconvex group lasso reg as a technique:
# reg: $\eta \times \Omega_g(\gamma)$ where $\Omega_g(\gamma) = \sum_{i,c} \norm{\gamma_{i,I_c}_1^{1/2}$
# I_c: index of samples from class $c$ in the source domain.
@instrument
def compute_all_wasserstein_from(src, user_distributions, M, sinkhorn_reg):
    try:
        return ot.gpu.sinkhorn(src, user_distributions.T, M, sinkhorn_reg)
    except Exception as e:
        logger.exception(e)

@instrument
def compute_all_wasserstein(user_distributions, M, sinkhorn_reg):
    ws = []
    for user_id in user_distributions:
        ws.append(compute_all_wasserstein_from(user_id, user_distributions, M, sinkhorn_reg))

    return np.vstack(ws)

def load_ot_data_in_gpu(input_ot_workload):
    with open(input_ot_workload, 'r') as f:
        _, C, user_distributions, _ = pickle.load(f)
        return ot.gpu.to_gpu(C, user_distributions)

def compute_squareform(cost_matrix):
    # FIXME: do it.
    pass

def main():
    parser = argparse.ArgumentParser(prog='gpu_sinkhorn', description='Run Sinkhorn on GPU over prepared OT data')
    # FIXME: more help.
    parser.add_argument('input_ot_workload')
    # TODO: Default to stdout, disable Python buffering
    parser.add_argument('output_wasserstein_matrix')
    parser.add_argument('-r', '--sinkhorn-regularization', default=1, type=float, description='Sinkhorn regularization')

    args = parser.parse_args()
    
    C, user_distributions = load_ot_data_in_gpu(args.input_ot_workload)
    # FIXME: add method (sqeuclidean, etc.)
    M = compute_squareform(C)
    # FIXME: add more parameters than just sinkhorn reg
    ws_matrix = compute_all_wasserstein(M, user_distributions, args.sinkhorn_regularization)
    # FIXME: check before if we have access, that's better to avoid obvious errors.
    with open(args.output_wasserstein_matrix, 'w') as f:
        pickle.dump(ws_matrix, f)

if __name__ == '__main__':
    main()
