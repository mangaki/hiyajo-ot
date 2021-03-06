#!/usr/bin/env python3
import numpy as np
import os
import logging

import argparse

from gpu_sinkhorn import load_ot_data_in_memory
from prepare_ot import load_ratings
from kernel_knn import KernelKNN, normalize

from utils import chrono, instrument
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

# Constants
KNN_RATING_VALUES = {
    "favorite": 1,
    "like": 1,
    "dislike": 0,
    "neutral": 0,
    "willsee": 1,
    "wontsee": 0,
}

METRICS = {
    'mse': mean_squared_error,
    'rmse': lambda yt, yp: mean_squared_error(yt, yp, squared=False)
}

logger = logging.getLogger()

def obvious_kernel(C):
    # we assume C is of the good size: i.e. C is of shape nb_works × embeddings_size where nb_works is the same as R.
    # we also assume that C is normalized.
    def inner_kernel(R, user_ids=None):
        if user_ids is not None:
            I = normalize(R[user_ids]) @ C
            J = (normalize(R) @ C).T
        else:
            I = normalize(R) @ C
            J = I.T

        return (I @ J)
    return inner_kernel

@instrument
def normalize_cost_matrix(C):
    with np.errstate(divide='ignore'):
        return C / C.max(axis=0)

def create_kernel_function(C):
    normalized_C = normalize_cost_matrix(C)
    etienne_kernel = obvious_kernel(normalized_C)
    return etienne_kernel

def not_readable_file(filename):
    return not os.path.isfile(filename) or not os.access(filename, os.R_OK)

def test_for_readability(filenames):
    for filename in filenames:
        if not_readable_file(filename):
            raise ValueError(
                    "{} is not reachable (permission denied or does not exist)".format(filename)
                    )

def append_measure(mes, filename):
    with open(filename, 'w') as f:
        f.write(str(mes) + '\n')

def start_measures(metric_name, filename):
    with open(filename, 'w') as f:
        f.write('\nMETRIC {}\n'.format(metric_name)) # TODO: add date

def measure(metric_name, y_pred, y_test):
    return METRICS[metric_name](y_test, y_pred)

def diag_imbalanced_info(y):
    labels = np.unique(y)
    counts = np.bincount(y.astype(np.int64))
    logger.info('Diagnostic of dataset imbalanced state')
    m, M = counts[0], counts[0]

    for label, count in zip(labels, counts):
        logger.info('{} label has {} element in the vector'.format(label, count))
        m = min(count, m)
        M = max(count, M)

    logger.info('[!] In summary, biggest deviation is: {}, corresponding to {} % of the maximum'.format(
        M - m,
        100 - 100*m/M
    ))

def main():
    parser = argparse.ArgumentParser(
            prog="etienne_knn",
            description="Run Etienne's KNN and cross-validate it over the y which was zipped in the prepared OT data")

    parser.add_argument(
            "initial_dataset",
            help="The initial dataset used"
            )
    parser.add_argument(
            "input_ot_workload",
            help="The OT dataset containing the cost matrix"
            )

    parser.add_argument("--chrono", action="store_true",
            help="Enable the chronometer")

    parser.add_argument("--save-results-to-file",
            help="Save the metrics results in the target filename")

    parser.add_argument("--metric",
            default='rmse',
            help="Metric name used for comparison, example: rmse")
    parser.add_argument("--shuffle-dataset",
            action='store_true',
            help='Shuffle the dataset through the folds and display the seed used')
    parser.add_argument("--auto-resize-cost-matrix",
            action='store_true',
            help='When nb_works > C.shape[0], C can be extended with zeros. Makes operations very slower.')
    parser.add_argument("--diagnose-balance-in-dataset",
            action='store_true',
            help='Show information about the balance in the dataset in terms of labels during folds and at start')
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increases the log verbosity for each occurrence",
    )
    parser.add_argument("--n-splits",
            default=5,
            type=int,
            help="The number of splits for the cross-validation process by N folds.")

    args = parser.parse_args()
    effectiveLevel = max(3 - max(args.verbose_count, int(args.chrono)), 0) * 10
    logger.setLevel(effectiveLevel)

    logging.info("Logging initialized at level: {}".format(effectiveLevel))

    chrono.is_enabled = args.chrono
    chrono.save("Arguments parsing")

    test_for_readability(filter(None,
        [args.input_ot_workload, os.path.join(args.initial_dataset, 'ratings.csv'), args.save_results_to_file]
        ))

    C, _ = load_ot_data_in_memory(args.input_ot_workload, True)
    _, X, y, nb_users, nb_works = load_ratings(args.initial_dataset, KNN_RATING_VALUES)
    
    if args.diagnose_balance_in_dataset:
        diag_imbalanced_info(y)

    chrono.save("OT data and dataset loaded in memory")

    if not C.shape[0] == nb_works and not args.auto_resize_cost_matrix:
        logger.warning("Cost matrix shape: {}, number of works in X: {}, X shape: {} ; errors will most likely happen.".format(C.shape, nb_works, X.shape))

    if args.auto_resize_cost_matrix:
        p_nb_works, embeddings_size = C.shape
        rC = np.zeros((nb_works, embeddings_size), dtype=np.uint8)
        rC[:p_nb_works,:embeddings_size] = C
        C = rC
        logger.info('Cost matrix resized: {}'.format(C.shape))

    if args.save_results_to_file:
        start_measures(args.metric, args.save_results_to_file)
        chrono.save("Measures file opened")

    etienne_kernel = create_kernel_function(C)
    chrono.save("Etienne's kernel built")

    if args.shuffle_dataset:
        r_state = np.random.randint(2**32)
        kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=r_state)
        logger.warning('Shuffling will create uncertainty in results, but here\'s the seed for the KFold: {}'.format(r_state))
    else:
        kf = StratifiedKFold(n_splits=args.n_splits)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        etienne_knn = KernelKNN(nb_users, nb_works, kernel_function=etienne_kernel)
        etienne_knn.fit(X_train, y_train)
        chrono.save("Etienne's KNN fitted")
       
        if args.diagnose_balance_in_dataset:
            logging.info('Imbalanced information on y_train')
            diag_imbalanced_info(y_train)
            logging.info('Imbalanced information on y_test')
            diag_imbalanced_info(y_test)

        y_pred = etienne_knn.predict(X_test)
        chrono.save("Etienne's KNN predicted")
        res = measure(args.metric, y_pred, y_test)
        logger.info('K-Fold [{}/{}]: metric={}'.format(i + 1, args.n_splits, res))
        chrono.save("Etienne's KNN measured")
        if args.save_results_to_file:
            append_measure(res, args.save_results_to_file)
        # FIXME: show some sort of confusion matrix (?) in debugging.

if __name__ == '__main__':
    main()
