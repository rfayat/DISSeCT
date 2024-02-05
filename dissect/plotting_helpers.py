"""A few utils functions for plotting inertial data.

Author: Romain FAYAT, February 2024"""
from cycler import cycler
import numpy as np
import warnings
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from .utils import\
    grab_current_axis, iterate_if_first_arg_is_iterable, grab_ax_boundaries
try:
    from angle_visualization import plot_projected_euclidean
except ImportError:
    warnings.warn("2D ", DeprecationWarning)
    

RGB = ["#D55E00", "#009E73", "#0072B2"]
GREEN, PURPLE = "#74b652", "#7b52ae"
RGB_CYCLER = cycler(color=RGB)

@grab_current_axis
@iterate_if_first_arg_is_iterable
def plot_line_vertical(x, ax=None, **kwargs):
    """Plot a vertical line with abscisse x.

    Giving an iterable of values as input will result in the plot of multiple
    vertical lines at the given locations.
    """
    _, y_limits = grab_ax_boundaries(ax)
    ax.plot([x, x], y_limits, **kwargs)
    ax.set_ylim(y_limits)
    return ax


@grab_current_axis
@iterate_if_first_arg_is_iterable
def plot_line_horizontal(y, ax=None, **kwargs):
    """Plot an horizontal line with ordinate y.

    Giving an iterable of values as input will result in the plot of multiple
    horizontal lines at the given locations.
    """
    x_limits, _ = grab_ax_boundaries(ax)
    ax.plot(x_limits, [y, y], **kwargs)
    ax.set_xlim(x_limits)
    return ax


@grab_current_axis
def plot_sphere(radius=1, origin=[0., 0., 0.], steps=10, ax=None, **kwargs):
    """Plot a 3D sphere on a 3D axis.
    
    kwargs are passed to ax.plot_surface
    """
    assert isinstance(ax, Axes3D)
    u, v = np.mgrid[0:2 * np.pi:2 * steps * 1j, 0:np.pi:steps * 1j]
    x = radius * np.cos(u) * np.sin(v) + origin[0]
    y = radius * np.sin(u) * np.sin(v) + origin[1]
    z = radius * np.cos(v) + origin[2]
    ax.plot_surface(x, y, z, **kwargs)
    return ax


@grab_current_axis
def plot_xyz(length=1., origin=np.array([0., 0., 0.]), ax=None, **kwargs):
    "Plot xyz vectors."
    for c, x in zip(RGB, np.eye(3)):
        ax.plot(*np.c_[origin, length * x + origin], c=c, **kwargs)
    return ax


def cmap_from_range(value, cmap, vmin=0., vmax=1.):
    "Rescale the range of a cmap using a vmin and vmax values."
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    value_new = cmap.N * (value - vmin) / (vmax - vmin)
    return cmap(int(value_new))


@grab_current_axis
def plot_gradient(X, ax=None, color=None, cmap=None, **kwargs):
    """Plot *X.T with a color gradient. Only valid in 2D.
    
    Inputs
    ------
    X: array, shape=(n, d)
        Values that will be plotted.
        
    ax: matplotlib axis, default=None
        Axis on which to plot, if None provided, grab the current axis.
        
    color: Valid color for plt.plot, default=None
        Color of the line. Ignored if cmap is provided.
        
    cmap: cmap or string compatible with sns.color_palette, default=None
        Cmap for the color gradient of the plot.
        
    **kwargs: Key-words arguments valid with plt.plot
    
    """
    if color is None:
        color = kwargs.pop("c", None)
    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)
    if cmap is not None:
        color = cmap(np.linspace(0, 1, len(X) - 1))
        
    S = np.concatenate([X[:-1, np.newaxis, :], X[1:, np.newaxis, :]], axis=1)
    if X.shape[1] == 2:
        L = LineCollection(S, color=color, **kwargs)
    elif X.shape[1] == 3:
        L = Line3DCollection(S, color=color, **kwargs)
    ax.add_collection(L)
    ax.autoscale()
    return ax


@grab_current_axis
def plot_projected_euclidean_gradient(x, ax=None, cmap="viridis", **kwargs):
    """Same as `plot_gradient` for `angle_visualization.plot_projected_euclidean`.
    
    Can be quite slow for large time series, if you find an equivalent to LineCollection
    generating similar plots (instead of the for loop), feel free to open a pull request.
    """
    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)
    for i in range(len(x) - 1):
        color = cmap_from_range(i, cmap, vmin=0, vmax=len(x) - 1)
        plot_projected_euclidean(*x[i:i+2].T, ax=ax, color=color, **kwargs)
    return ax
