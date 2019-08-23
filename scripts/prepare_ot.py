#!/usr/bin/env python
from itertools import chain
import argparse
import numpy as np
from zero import MangakiKNN
from zero.chrono import Chrono
from datetime import datetime
import pickle
import logging
import os
import pandas as pd

from embeddings import generate_mapping, merge_and_order_embeddings
from utils import chrono, instrument

# Logger
logger = logging.getLogger()

# Constants
RATING_VALUES = {'favorite': 1, 'like': 1, 'dislike': 0, 'neutral': 0,
                 'willsee': 0, 'wontsee': 0}

@instrument
def load_rating_values_file(path):
    mandatory_fields = {'favorite', 'like', 'dislike', 'neutral', 'willsee', 'wontsee'}
    try:
        with open(path, 'r') as f:
            local_rating_values = json.loads(f.read())
            for field in mandatory_fields:
                if field not in local_rating_values:
                    raise ValueError('Incorrect local rating values file, missing field: {} !'.format(field))
            
            RATING_VALUES = local_rating_values
    except Exception as e:
        logger.exception(e)
        logger.info('Continuing with binary values: {}'.format(RATING_VALUES))


@instrument
def warn_for_sequential_ids_mismatches(embeddings):
    # Check for potential mismatches
    A = set(sorted(list(chain(*[embedding.work_ids for embedding in embeddings]))))
    B = set(range(1, max(A) + 1))
    
    n_samples = len(A ^ B)
    if n_samples > 0:
        logger.warning('There are at least {} missing embeddings, the results might be wrong.'.format(n_samples))

@instrument
def warn_for_size_mismatches(embeddings):
    for embedding in embeddings:
        n_samples, _ = embedding.npy.shape
        n_ids = len(embedding.work_ids)

        if n_samples != n_ids:
            logger.warning('There are {} samples in {} and {} work IDs fetched from the path file'.format(
                n_samples,
                embedding.npy_filename,
                n_ids))

@instrument
def load_ratings(path):
    df = pd.read_csv(os.path.join(path, 'ratings.csv'))
    triplets = np.array(df[['user', 'item', 'rating']], dtype=np.object)
    vectorized_convert = np.vectorize(RATING_VALUES.get, otypes=[np.float64])
    X = triplets[:,0:2].astype(np.int32)
    y = vectorized_convert(triplets[:,2])
    nb_users = int(max(triplets[:,0]) + 1)
    nb_works = int(max(triplets[:,1]) + 1)
    
    return df, X, y, nb_users, nb_works

@instrument
def open_embeddings(data_path: str):
    embeddings = generate_mapping(data_path, False)
    if not embeddings:
        raise ValueError('No embeddings, no computations')

    for embedding in embeddings:
        embedding.open_npy()

    warn_for_sequential_ids_mismatches(embeddings)
    warn_for_size_mismatches(embeddings)
    return embeddings

@instrument
def filter_ratings(ratings_path: str, user_threshold: int = 100):
    _, X, y, nb_users, nb_works = load_ratings(ratings_path)
    
    # user_threshold = 0 ⇒ no threshold.
    if user_threshold > 0:
        user_filter = X[:,0] <= user_threshold
        X, y = X[user_filter], y[user_filter]
        nb_users = X[:,0].max() + 1

    return X, y, nb_users, nb_works

@instrument
def compute_mask_based_on(base_work_ids, target_work_ids, n_samples):
    index, = np.where(~np.isin(target_work_ids, base_work_ids))
    mask = np.ones(n_samples, np.bool)
    mask[index] = 0
    return index, mask

@instrument
def filter_ratings_based_on_cost_matrix_encoder(X, y, encoder, sanity_check: bool = False):
    work_ids = list(encoder.encoder.keys())
    index, mask = compute_mask_based_on(work_ids, X[:, 1], X.shape[0])

    if sanity_check:
        u = set(work_ids)
        assert all((i not in u for _, i in X[index]))
        assert all((i in u for _, i in X[mask]))

    # filter using the mask.
    X, y = X[mask], y[mask]

    # re-encode our work IDs to match the cost matrix indexes.
    X[:,1] = np.asarray(list(map(encoder.encode, X[:,1])))

    nb_works = X[:,1].max() + 1
    nb_users = X[:,0].max() + 1
    # X, y, nb_users, nb_works
    return X, y, nb_users, nb_works

@instrument
def filter_ratings_based_on_cost_matrix(C, X, y, encoder, sanity_check: bool = False):
    X, y, nb_users, nb_works = filter_ratings_based_on_cost_matrix_encoder(X, y, encoder, sanity_check)
    C = C[:nb_works]

    logger.info('Cost matrix (final) shape: {}'.format(C.shape))
    logger.info('Ratings has {} users, {} works, X has shape {}, y has shape {}'.format(nb_users, nb_works,
        X.shape,
        y.shape))

    return C, X, y, nb_users, nb_works

@instrument
def create_cost_matrix(X, y, embeddings, sanity_check: bool = False):
    C, encoder = merge_and_order_embeddings(embeddings)
    C, X, y, nb_users, nb_works = filter_ratings_based_on_cost_matrix(C, X, y, encoder, sanity_check)
    return C, encoder, X, y, nb_users, nb_works

@instrument
def compute_user_distribution(user_id, items_matrix, method='divide', epsilon=1e-2):
    A = items_matrix[user_id].todense() + epsilon

    if method == 'softmax':
        A = softmax(A)
    elif method == 'divide':
        A /= A.sum()
    else:
        assert callable(method), 'The method to compute user distribution must be a callable if not softmax or divide'
        A = method(A)

    return np.asarray(A).reshape(-1)

@instrument
def compute_user_distributions(nb_users, items_matrix, method='divide', epsilon=1e-2):
    local_cud = lambda user_id: compute_user_distribution(user_id, items_matrix, method, epsilon)
    return np.asarray(list(map(local_cud, range(1, nb_users))))

def main():
    parser = argparse.ArgumentParser(prog='prepare_ot', description='Prepare data for Optimal Transport computations')
    # Mandatory
    parser.add_argument('data_path', type=str, help='Path to the data directory (ratings, embeddings)')
    parser.add_argument('--output', default='ot.npy', help='Output file consisting of (knn.M original matrix, cost matrix, user distributions matrix, item encoder), default: ot.npy')
    # Utilities
    parser.add_argument('--chrono', action='store_true', help='Enable the chronometer')
    parser.add_argument('--sanity-check', action='store_true', help='Run the sanity check (assertions, it is *SLOWER*)')
    parser.add_argument('-v', '--verbose', dest='verbose_count', action='count',
            default=0, help='Increases the log verbosity for each occurrence')
    # Rating values
    parser.add_argument('-lrf', '--load-rating-values-file', help='Load a JSON rating values file, see an example in README.md (default to binary values)')
    # Threshold controllers
    # TODO: add work threshold also.
    parser.add_argument('--user-threshold', type=int, default=100, help='Limit the amount of users loaded from ratings (default: 100, use 0 for +∞)')
    # User distribution scheme computation
    parser.add_argument('--method', type=str, default='divide', help='Method to use to compute the user distribution from KNN matrix (default: divide by sum)')
    parser.add_argument('--epsilon', type=float, default=1e-2, help='Epsilon to perturb an user distribution (lower makes the result more exact, default: 1e-2)')
    # KNN parameters
    parser.add_argument('--knn-neighbors', type=int, default=20, help='Number of neighbors for MangakiKNN instance (default: 20)')
    parser.add_argument('--knn-rated-by-neighbors-at-least', type=int, default=3, help='Rated by neighbors at least parameter for MangakiKNN instance (default: 3)')
    parser.add_argument('--knn-weighted-neighbors', action='store_true', help='Enable weighted neighbors for MangakiKNN instance')

    
    args = parser.parse_args()

    effectiveLevel = max(3 - max(args.verbose_count, int(args.chrono)), 0) * 10
    logger.setLevel(effectiveLevel)

    logging.info('Logging initialized at level: {}'.format(effectiveLevel))


    chrono.is_enabled = args.chrono
    chrono.save('Argument parsing')

    if args.load_rating_values_file:
        load_rating_values_file(args.load_rating_values_file)

    if os.path.isfile(args.output) and not os.access(args.output, os.W_OK):
        raise ValueError('Output file cannot be accessed in write mode, aborting the preparation')

    embeddings = open_embeddings(args.data_path)
    X, y, nb_users, nb_works = filter_ratings(args.data_path, args.user_threshold)
    C, encoder, X, y, nb_users, nb_works = create_cost_matrix(X, y, embeddings, args.sanity_check)
    chrono.save('Data loaded in memory')
    knn = MangakiKNN(nb_neighbors=args.knn_neighbors,
            rated_by_neighbors_at_least=args.knn_rated_by_neighbors_at_least,
            weighted_neighbors=args.knn_weighted_neighbors)
    knn.set_parameters(nb_users, nb_works)
    knn.fit(X, y)
    chrono.save('KNN trained')
    user_distributions = compute_user_distributions(nb_users, knn.M, args.method, args.epsilon)
    chrono.save('User distributions computed')
    if args.sanity_check:
        logger.info('knn.M[{}]: {}'.format(knn.M.shape, knn.M))
        logger.info('C[{}]: {}'.format(C.shape, C))
        logger.info('User distributions[{}]: {}'.format(user_distributions.shape, user_distributions))
    try:
        # write (knn.M, C, user_distributions, encoder) into output
        with open(args.output, 'wb') as f:
            raw_repr = pickle.dumps((knn.M,
                C,
                user_distributions,
                encoder), pickle.HIGHEST_PROTOCOL)

            f.write(raw_repr)
        chrono.save('Wrote final data on disk')
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
    main()
