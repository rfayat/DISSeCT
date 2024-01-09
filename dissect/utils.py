"A few utils routine."
from joblib import Parallel, delayed
from functools import wraps
import pandas as pd
import numpy as np


def split_and_map(X, split_index, func, n_jobs=None):
    """Split X at indices split_index and map func on the result.

    If n_jobs is not None, computations are performed in parallel.
    """
    X_split = np.split(X, split_index)
    if n_jobs is None:
        return [*map(func, X_split)]
    return Parallel(n_jobs=n_jobs)(delayed(func)(x) for x in X_split)


def agglomerate(X, by, agg_func="mean", return_categories=False):
    "Agglomerate values of X by a set of categories."
    d = pd.DataFrame(X, index=pd.Index(by, name="by"))
    d.reset_index(drop=False)
    agglomerated = d.groupby(by="by").agg(agg_func)
    if return_categories:
        return agglomerated.values, agglomerated.index.values
    else:
        return agglomerated.values


def reshape_input_2D(f):
    "Add sanity check on 2D array input"
    @wraps(f)
    def g(X, *args, **kwargs):
        if isinstance(X, np.ndarray) and X.ndim != 2:
            X = X.reshape(-1, 1)
        return f(X, *args, **kwargs)
    return g


def handle_col_names(f):
    "Grab the column names from a DataFrame or input."
    @wraps(f)
    def g(X, *args, columns=None, **kwargs):
        if isinstance(X, pd.Series):
            columns = [X.name]
            X = X.values
        elif isinstance(X, pd.DataFrame):
            columns = list(X)
            X = X.values
        elif columns is None:
            columns = [f"{i}" for i in range(X.shape[1])]
        else:
            try:
                assert len(columns) == X.shape[1]
            except AssertionError:
                raise ValueError("columns must have the same length as the"
                                 " number of columns in X")
        return f(X, *args, columns=columns, **kwargs)

    return g
