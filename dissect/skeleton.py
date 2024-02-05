"""Utils functions for plotting 3D pose estimate.

Designed to deal with coordinate format obtained using anipose.

Author: Romain FAYAT, February 2024"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from typing import List
from .utils import grab_current_axis
import seaborn as sns
from scipy.spatial.transform import Rotation as R
import re

# Hard-coded joints for the pose
SCHEME = [
    ["snout", "earBaseR"],
    ["snout", "earBaseL"],
    ["earBaseR", "earBaseL"],
    ["earBaseR", "neck"],
    ["earBaseL", "neck"],
    ["snout", "neck"],
    ["neck", "forePawR"],
    ["neck", "forePawL"],
    ["neck", "middleSpine"],
    ["middleSpine", "tailBase"],
    ["tailBase", "tailMiddle"],
    ["tailMiddle", "tailTip"],
    ["tailBase", "hindPawR"],
    ["tailBase", "hindPawL"],
]
# Color gradient for skeleton plotting
BONE_COLOR_IDX = {'snout-earBaseR': 0,
                  'snout-earBaseL': 0,
                  'earBaseR-earBaseL': 1,
                  'earBaseR-neck': 1,
                  'earBaseL-neck': 1,
                  'snout-neck': 1,
                  'neck-forePawR': 2,
                  'neck-forePawL': 2,
                  'neck-middleSpine': 3,
                  'middleSpine-tailBase': 3,
                  'tailBase-tailMiddle': 4,
                  'tailMiddle-tailTip': 4,
                  'tailBase-hindPawR': 5,
                  'tailBase-hindPawL': 5}

BODYPART_COLOR_IDX = {'forePawR': 2,
                      'hindPawR': 5,
                      'forePawL': 2,
                      'earBaseR': 1,
                      'tailMiddle': 4,
                      'snout': 0,
                      'tailBase': 4,
                      'earBaseL': 1,
                      'middleSpine': 3,
                      'hindPawL': 5,
                      'tailTip': 4,
                      'neck': 3}


def bodyparts_from_scheme(scheme: List[List[str]]):
    "Return unique bodypart names from pairs of bodyparts."
    # Flatten the nested list of bodyparts
    bodyparts = []
    for pair in scheme:
        bodyparts = [*bodyparts, *pair]
    # Return the unique elements of the list
    return list(set(bodyparts))


def get_points(pose_frame: pd.Series, bodyparts: List[str]):
    "Get the xyz coordinates of each bodypart and return them as an array."
    coord = np.zeros((len(bodyparts), 3))
    for i, part in enumerate(bodyparts):
        col_names = [f"{part}_{ax}" for ax in "xyz"]
        coord[i, :] = pose_frame[col_names].values
    return coord


def get_skeleton(pose_frame: pd.Series, scheme: List[List[str]]):
    """Return pairs of coordinates of points from a skeleton scheme.

    The input `scheme` contains `n_bone` pairs of bodyparts as strings.

    Returns a (n_bone, 2, 3) array of the start and end 3d coordinates
    of the bodypart for each 'bone'.
    """
    # Get the extremities of each bone
    bodypart_start = [e[0] for e in scheme]
    coord_start = get_points(pose_frame, bodypart_start)
    bodypart_end = [e[1] for e in scheme]
    coord_end = get_points(pose_frame, bodypart_end)
    return np.stack((coord_start, coord_end), axis=1)


def translate_pose(pose, x=0, y=0, z=0, scheme=SCHEME):
    "Translate a pose time series."
    pose_new = pose[["chunk", "fnum"]].copy()

    for pattern, offset in zip(["_x", "_y", "_z"], [x, y, z]):
        col_axis = [p for p in list(pose) if re.search(pattern, p)]
        for c in col_axis:
            pose_new[c] = pose[c].values.copy() + offset

    return pose_new


def rotate_pose(pose, rot, scheme=SCHEME):
    "Rotate a pose time series."
    bodyparts = bodyparts_from_scheme(scheme)
    pose_new = pose[["chunk", "fnum"]].copy()
    for bp in bodyparts:
        bp_cols = [f"{bp}_{axis}" for axis in "xyz"]
        bp_coords_new = rot.apply(pose[bp_cols].values)
        for c, coords in zip(bp_cols, bp_coords_new.T):
            pose_new[c] = coords.copy()
    return pose_new


def center_pose(pose, scheme=SCHEME, how="first"):
    """Translate and rotate pose time series obtain a centered skeleton.
    
    The `how` argument can take values "first", "last", "middle" or an integer
    indicating which skeleton to use as a reference and corresponds to the
    skeleton used as a reference to center and rotate all provided poses.
    
    Note
    ----
    We will use the tailbase-neck azimuth in the earth reference frame for the
    azimuth and the tailbase coordinates for the xy coordinates.
    
    """
    # Select the pose that will be used as a reference
    if how == "first":
        iloc_skeleton = 0
    elif how == "last":
        iloc_skeleton = len(pose) - 1
    elif how == "middle":
        iloc_skeleton = len(pose) // 2
    elif isinstance(how, int):
        iloc_skeleton = how
    else:
        raise ValueError("`how` must be 'first', 'last', 'middle' or an int.")
        
    # Translation
    pose = translate_pose(pose, x=-pose.iloc[iloc_skeleton].tailBase_x,
                          y=-pose.iloc[iloc_skeleton].tailBase_y,
                          scheme=scheme)

    # Rotation
    x_bone = pose.iloc[iloc_skeleton].neck_x - pose.iloc[iloc_skeleton].tailBase_x
    y_bone = pose.iloc[iloc_skeleton].neck_y - pose.iloc[iloc_skeleton].tailBase_y
    azim = np.arctan2(y_bone, x_bone)
    rot = R.from_euler("zxy", [-azim, 0, 0])
    return rotate_pose(pose, rot, scheme=scheme)


@grab_current_axis
def plot_skeleton(pose_frame, scheme=SCHEME, ax=None,
                  kwargs_scatter=None, kwargs_plot=None):
    "Plot a skeleton from point coordinates and bone name pairs."
    # Make sure the axis is 3-dimensional
    assert isinstance(ax, Axes3D)

    # Replace plotting kwargs by empty dictionary if None are provided
    if kwargs_scatter is None:
        kwargs_scatter = {}
    if kwargs_plot is None:
        kwargs_plot = {}

    # Grab the coordinates of the bodyparts and 'bones' to plot
    bodyparts_to_plot = bodyparts_from_scheme(scheme)
    bodyparts_coord = get_points(pose_frame, bodyparts_to_plot)
    skeleton_coord = get_skeleton(pose_frame, scheme)
    # Plot the bodyparts and the skeleton
    ax.scatter(*bodyparts_coord.T, **kwargs_scatter)
    for bone_coord in skeleton_coord:
        ax.plot(*bone_coord.T, **kwargs_plot)
    return ax


@grab_current_axis
def plot_skeleton_gradient(pose_frame, ax=None, kwargs_scatter=None,
                           kwargs_plot=None, cmap="autumn_r", scheme=SCHEME):
    "Plot a skeleton using an antero posterior color gradient."
    # Make sure the axis is 3-dimensional
    assert isinstance(ax, Axes3D)

    # Grab the coordinates of the bodyparts and 'bones' to plot
    bodyparts_to_plot = bodyparts_from_scheme(scheme)
    bodyparts_coord = get_points(pose_frame, bodyparts_to_plot)
    skeleton_coord = get_skeleton(pose_frame, scheme)

    # Grab the color of each "bone" and joint
    colors = sns.color_palette(cmap, max(BONE_COLOR_IDX.values()) + 1)
    skeleton_colors = [colors[BONE_COLOR_IDX["-".join(e)]] for e in scheme]
    bodypart_colors = [colors[BODYPART_COLOR_IDX[e]]
                       for e in bodyparts_to_plot]

    # Replace plotting kwargs by empty dictionary if None are provided
    if kwargs_scatter is None:
        kwargs_scatter = {}
    if kwargs_plot is None:
        kwargs_plot = {}

    # Plot the bodyparts and the skeleton
    for bp, c in zip(bodyparts_coord, bodypart_colors):
        ax.scatter(*bp, c=[c], **kwargs_scatter)
    for bone_coord, c in zip(skeleton_coord, skeleton_colors):
        ax.plot(*bone_coord.T, c=c, **kwargs_plot)
    return ax


@grab_current_axis
def plot_skeleton_gradient_timeseries(pose, ax=None, kwargs_scatter=None,
                                      kwargs_plot=None, cmap="autumn_r",
                                      alpha_vmin=.1, alpha_vmax=.4, scheme=SCHEME):
    """Overlay skeletons stored in a dataframe using a transparency gradient.
    
    (Very heavily) inspired by Weinreb et al. 2023, figure 6k.
    """
    if kwargs_plot is None:
        kwargs_plot = {}
    if kwargs_scatter is None:
        kwargs_scatter = {}

    n_pose = len(pose)
    alpha_all = np.append(np.linspace(alpha_vmin, alpha_vmax, n_pose - 1), 1)

    for alpha, (_, pose_frame) in zip(alpha_all, pose.iterrows()):
        plot_skeleton_gradient(pose_frame, ax=ax,
                               kwargs_scatter={
                                   **kwargs_scatter, "alpha": alpha},
                               kwargs_plot={**kwargs_plot, "alpha": alpha},
                               cmap=cmap, scheme=scheme)
    return ax
