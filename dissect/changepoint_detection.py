"""Parallel computation of changepoints and  segmentation quality metrics.

Author: Romain FAYAT, April 2022
"""
import numpy as np
import ruptures as rpt
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score


def chpt_to_label(bkps):
    """Return the segment index each sample belongs to.

    Example:
    -------
    >>> chpt_to_label([4, 10])
    array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    """
    duration = np.diff([0] + bkps)
    return np.repeat(np.arange(len(bkps)), duration)


def randindex_adjusted(bkps1, bkps2):
    "Compute the adjusted rand index for two lists of changepoints."
    label1 = chpt_to_label(bkps1)
    label2 = chpt_to_label(bkps2)
    return adjusted_rand_score(label1, label2)


def approximate_gamma(X, n_max=10000):
    """Approximate the median heuristic for rbf kernel by downsampling X.

    cf code from ruptures.costs.costrbf
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(X)
    # Keep only maximum n_max points
    if n > n_max:
        idx_selected = np.random.choice(np.arange(n), n_max, replace=False)
        X = X[idx_selected]
    # Compute the invert median heuristic for the rbf bandwidth parameter
    K = pdist(X, metric="sqeuclidean")
    K_median = np.median(K)
    return 1. / K_median if K_median != 0 else 1.


def kernelcpd(X, pen, kernel="linear", min_size=2, params=None):
    "Compute kernel changepoint detection on X."
    estimator = rpt.KernelCPD(kernel=kernel, min_size=min_size, params=params)
    return np.array(estimator.fit_predict(X, pen=pen))


def kernelcpd_parallel(X, pen, n_points_chunk=10000, kernel="linear",
                       min_size=2, params=None, n_jobs=-1):
    """Compute kernel CPD in a parallel fashion using chunks of X.

    Example:
    -------
    X = np.random.random(100000).reshape(50000, 2)
    print(kernelcpd_parallel(X, pen=2, kernel="rbf",
                             n_points_chunk=1000, n_jobs=-1))

    """
    # Use the same value for the bandwith of the Gaussian kernel if needed
    if kernel == "rbf" and (params is None or params.get("gamma") is None):
        params = {"gamma": approximate_gamma(X)}
    # Perform CPD on chunks with max length n_points_chunk
    X_split = np.split(X, np.arange(n_points_chunk, len(X), n_points_chunk))
    kw = dict(pen=pen, kernel=kernel, min_size=min_size, params=params)
    chpt_all = Parallel(n_jobs=n_jobs)(
        delayed(lambda x: kernelcpd(x, **kw))(x) for x in X_split
    )
    # Combine the detected changepoints
    for i, chpt in enumerate(chpt_all):
        chpt_all[i] = chpt[:-1] + i * n_points_chunk
    chpt_all[-1] = np.append(chpt_all[-1], len(X))
    return np.concatenate(chpt_all).tolist()


if __name__ == "__main__":
    X = np.random.random(100000).reshape(50000, 2)
    print(kernelcpd_parallel(X, pen=2, kernel="rbf",
                             n_points_chunk=1000, n_jobs=-1))
