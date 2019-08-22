# How to use those scripts?

Basically: `prepare_ot.py /path/to/embeddings_and_ratings | gpu_sinkhorn.py > wasserstein_matrix`.

In fact, it's more: `prepare_ot.py somewhere --output ot.npy && gpu_sinkhorn.py ot.npy wasserstein_matrix` right now.

## Prepare OT script

Prepare OT script expects a directory where you have the `embeddings-*` file and the `ratings.csv` file at the same time, it'll then write an `ot.npy` file by default containing: `knn.M` matrix, cost matrix (nb_works × 512 in general), user distributions matrix (nb_users × nb_works), an encoder from ratings work IDs to matrices indexes (useful if you want to know what index is what anime in the rating CSV if you have titles, that is debugging).

This file is a pickle dump of a tuple of `np.array`.

## GPU Sinkhorn

It reads the `ot.npy` file in the format explained before, upload it to GPU, run GPU Sinkhorn on every couple of users, merge everything and spit back a matrix of all pairwise Wasserstein distances between users.

The output file is a pickle dump of the GPU array (!).

## In case of bugs

Use verbosity at max and enable chronometer.
