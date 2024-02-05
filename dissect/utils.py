"A few utils routine."
from joblib import Parallel, delayed
from functools import wraps
import pandas as pd
import numpy as np
from typing import Iterable
from functools import wraps
import matplotlib.pyplot as plt

## -----------Matplotlib utils-----------


def is_iterable(obj):
    "Check if an object is an iterable."
    return isinstance(obj, Iterable)


def iterate_if_first_arg_is_iterable(f):
    "Add the support of iterating over the first arg to a function."
    @wraps(f)
    def g(X, *args, **kwargs):
        if is_iterable(X):
            return [f(x, *args, **kwargs) for x in X]
        else:
            return f(X, *args, **kwargs)
    return g


def grab_current_axis(f):
    "Grab the current axis if None is provided."
    @wraps(f)
    def g(*args, **kwargs):
        "Call f after grabbing the current axis if None is provided."
        if kwargs.get("ax") is None:
            kwargs.update({"ax": plt.gca()})
        return f(*args, **kwargs)

    return g


def grab_ax_boundaries(ax):
    "Find the axis boundaries."
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    return x_limits, y_limits


@grab_current_axis
def equalize_xy(ax):
    "Make the xlim and ylim of ax the same amplitude."
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    width = max(ylim[1] - ylim[0], xlim[1] - xlim[0]) / 2
    xlim_center = (xlim[0] + xlim[1]) / 2
    ylim_center = (ylim[0] + ylim[1]) / 2
    ax.set_xlim([xlim_center - width, xlim_center + width])
    ax.set_ylim([ylim_center - width, ylim_center + width])
    return ax


## -----------Pandas / numpy utils-----------


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
