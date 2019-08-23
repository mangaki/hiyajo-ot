# How to use those scripts?

Basically: `prepare_ot.py /path/to/embeddings_and_ratings | gpu_sinkhorn.py > wasserstein_matrix`.

In fact, it's more: `prepare_ot.py somewhere --output ot.npy && gpu_sinkhorn.py ot.npy wasserstein_matrix` right now.

## Dependencies

### Prepare OT script

NumPy, SciPy, Mangaki/Zero.

### GPU Sinkhorn

NumPy, SciPy, for the GPU version: CuPy and a CUDA core (GPU/GPGPU whatever).

**WARNING** : You have to have the CUDA toolkit and `pip` install the good cupy if you don't want to compile it. (`cupy-cuda101` for example for CUDA 10.1, it is a pre-compiled wheel).

## Prepare OT script

Prepare OT script expects a directory where you have the `embeddings-*` and `paths-*` files and the `ratings.csv` file at the same time, it'll then write an `ot.npy` file by default containing: `knn.M` matrix, cost matrix (nb_works × 512 in general), user distributions matrix (nb_users × nb_works), an encoder from ratings work IDs to matrices indexes (useful if you want to know what index is what anime in the rating CSV if you have titles, that is debugging).

You can use threshold to control the size of the output dataset (`--user-threshold` and `--work-threshold`), it'll automatically re-encode if necessary the IDs and output a correct dataset, the encoder being given, you can inspect back your final results using the encoder.

You can also use a local rating values using the `--load-rating-values-file` or `-lrf` flag and giving it a JSON file of the form: `{"favorite": <float>, "like:" <float>, "dislike": <float>, "neutral": <float>, "willsee": <float>, "wontsee": <float>}` --- all these keys must be present, otherwise it will fallback to a binary rating values mapping.

This file is a pickle dump of a tuple of `np.array`.

**WARNING** : It is expected for `paths-*` file to contain a path which ends in the **WORK ID** of the currently processed work.

## GPU Sinkhorn

It reads the `ot.npy` file in the format explained before, upload it to GPU, run GPU Sinkhorn on every couple of users, merge everything and spit back a matrix of all pairwise Wasserstein distances between users.

The output file is a pickle dump of the GPU array (!).

## In case of bugs

### General

Use verbosity at max and enable chronometer, enable sanity checks.

### GPU Sinkhorn

Run it using CPU fallback with `--use-cpu`.
