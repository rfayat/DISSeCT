"""Feature extraction from inertial data.

Author: Romain FAYAT, February 2022
"""
import pandas as pd
import numpy as np
from SphereProba.distributions import VonMisesFisher, _fit_kent
import scipy.stats
from scipy.spatial.transform import Rotation as R
from dissect.data_loading import COL_ACC_G, COL_GYR, COL_QUAT, COL_ACC_R, SR
from dissect.angles import get_angle, get_azimuth_estimate
from sklearn.linear_model import LinearRegression, HuberRegressor
from dissect.utils import reshape_input_2D, handle_col_names, split_and_map


def gradient_df(df):
    "Return the gradient of each column of a dataframe, preserving the DataFrame."
    grad = np.gradient(df.values, axis=0)
    col_new = [f"{c}_gradient" for c in df.columns]
    return pd.DataFrame(grad, columns=col_new, index=df.index)


def tilt_from_euclidean(X, degrees=True):
    """Return roll and pitch from the euclidean coordinates of X.

    Not the usual definition of these angles, roll angle is here the smallest
    angle to the xz plane (in -90, 90) while pitch is between -180 and 180.

    North pole corresponds to 0 for both.

    Parameters
    ----------
    X : array, shape = (n_points, 3)
        The euclidean coordinates on S2
    degrees : bool (default: True)
        Set `degrees`to True to obtain values in degrees, False for radians.

    """
    theta1 = np.arctan2(X[:, 1], X[:, 2])
    theta2 = np.arctan2(X[:, 1], -X[:, 2])
    select_theta1 = np.abs(theta1) < np.abs(theta2)
    roll = np.where(select_theta1, theta1, theta2)
    pitch = np.arctan2(X[:, 0], X[:, 2])

    if degrees:
        return np.degrees(np.c_[roll, pitch])
    else:
        return np.c_[roll, pitch]


def extract_vmf_features(X, prepend=""):
    "Features extracted by fitting a vMF distribution on X."
    vmf = VonMisesFisher.fit(X)
    dispersion = np.sqrt(1 / vmf.kappa)
    features_names = ["dispersion", "mux", "muy", "muz"]
    features = [dispersion, *vmf.mu.squeeze()]
    return {f"{prepend}{k}": f for k, f in zip(features_names, features)}


def extract_kent_features(X, prepend=""):
    """Features extracted by fitting a Kent distribution on X.
    
    Returns
    -------
    roll :
        Roll of the distribution's centroid (in degrees)
    pitch :
        Roll of the distribution's centroid (in degrees)
    theta :
        Angle between the ellipsoid's main axis and the vertical (in degrees) 
    kappa :
        Dispersion of the distribution
    beta :
        Ellipticity of the distribution

    Notes
    -----
    We use _fit_kent instead of the Kent object to bypass the assertion on
    the fitted parameters. The approximation used in SphereProba (Kent 1982)
    indeed holds for kappa>>beta. In our case, we will get large values of
    beta for narrow distributions (e.g. arc on the sphere). The parameter
    estimate is therefore not very accurate but still will yield parameters
    reflecting the elongation of the ellipsoid on the sphere.

    """
    gamma, kappa, beta = _fit_kent(X, np.ones(len(X)))
    mu = gamma[[0]]
    roll, pitch = tilt_from_euclidean(mu).squeeze()
    theta = get_angle(gamma[1], np.array([0., 0., 1.]))

    kappa = np.clip(kappa, 1e-6, None)
    beta = np.clip(beta, 1e-6, None)

    features_names = ["roll", "pitch", "theta", "logkappa", "logbeta"]
    features = [roll, pitch, theta, np.log(kappa), np.log(beta)]
    return {f"{prepend}{k}": f for k, f in zip(features_names, features)}


@reshape_input_2D
@handle_col_names
def compute_corr(X, columns=None):
    "Return a dictionary of the correlations of the different axes."
    corr_all = {}
    corrcoef = np.corrcoef(X, rowvar=False)
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            if i > j:  # Lower diagonal of the correlation matrix
                corr_all[f"corr_{c1}_{c2}"] = corrcoef[i, j]
    return corr_all


@reshape_input_2D
@handle_col_names
def compute_autoregressive_coef(X, columns=None, fit_intercept=True, robust=False, **kwargs):
    """Coefficients for an AR model with lag 1.
    
    Returns the coefficient (elements of A) of a linear model with form:
        X_{t+1} = X_{t} + A.X_{t}
        
    DO NOT USE as features directly, useful for the singular values of the matrix ? 
    """
    if robust:  # We need one regressor per dimension (only 1-D regression is supported)
        # N.B. Maybe not a good idea because we fit different intercepts
        coef = []
        for i, x in enumerate(X.T):
            lr = HuberRegressor(fit_intercept=fit_intercept,
                                **kwargs).fit(X[:-1], x[1:])
            coef.append(lr.coef_)
        coef = np.vstack(coef).T

    else:
        lr = LinearRegression(fit_intercept=fit_intercept).fit(X[:-1], X[1:])
        coef = lr.coef_
    ar_coef = {}
    for i, c1 in enumerate(columns):
        for j, c2 in enumerate(columns):
            ar_coef[f"AR_{c1}_{c2}"] = coef[i, j]
    return ar_coef


@reshape_input_2D
@handle_col_names
def compute_basic_features(X, columns=None, compute_quartiles=True, compute_higher_moments=True):
    "Basic 1D features for each axis of a multi-dimensional time series."
    mu = {f"{c}_mean": m for c, m in zip(columns, X.mean(axis=0))}
    sigma = {f"{c}_std": m for c, m in zip(columns, X.std(axis=0))}
    min_all = {f"{c}_min": m for c, m in zip(columns, X.min(axis=0))}
    max_all = {f"{c}_max": m for c, m in zip(columns, X.max(axis=0))}

    quartiles = {}
    if compute_quartiles:
        # quartiles of each column
        quartiles_all = np.percentile(X, [25., 50., 75.], axis=0)
        for p_all, name in zip(quartiles_all, ["p25", "median", "p75"]):
            for p, c in zip(p_all, columns):
                quartiles[f"{c}_{name}"] = p

    higher_moments = {}
    if compute_higher_moments:
        skew_all = scipy.stats.skew(X, axis=0)
        kurt_all = scipy.stats.kurtosis(X, axis=0)
        for c, skew, kurt in zip(columns, skew_all, kurt_all):
            higher_moments[f"{c}_skew"] = skew
            higher_moments[f"{c}_kurt"] = kurt

    return {**mu, **sigma, **min_all, **max_all, **quartiles, **higher_moments}


def number_crossing_m(x, m=0.):
    "Number of crossing of value m by the 1D array x."
    is_crossing_up = (x[:-1] <= m) & (x[1:] > m)
    is_crossing_down = (x[:-1] > m) & (x[1:] < m)
    return is_crossing_up.sum() + is_crossing_down.sum()


@reshape_input_2D
@handle_col_names
def compute_ts_features(X, columns=None):
    "Basic time-series 1D features for each axis of a multi-dimensional time series."
    ts_features = {}
    for x, c in zip(X.T, columns):
        ts_features[f"{c}_zerocrossing"] = number_crossing_m(x, 0.)
    return ts_features


def compute_tilt_features(X, sr=SR):
    "Features derived from tilt (accG)."
    duration = len(X) / sr
    # Kent features for tilt
    vmf_attitude = extract_vmf_features(X, prepend="attitude_")

    # Tilt change between the 1st and last sample
    tilt_change_angle = get_angle(X[0], X[-1])
    tilt_change_speed = tilt_change_angle / duration

    return {
        **vmf_attitude,
        "tilt_change_x": X[-1, 0] - X[0, 0],
        "tilt_change_y": X[-1, 1] - X[0, 1],
        "tilt_change_z": X[-1, 2] - X[0, 2],
        "tilt_change_angle": tilt_change_angle,
        "tilt_change_speed": tilt_change_speed,
        "tilt_start_x": X[0, 0],
        "tilt_start_y": X[0, 1],
        "tilt_start_z": X[0, 2],
        "tilt_end_x": X[-1, 0],
        "tilt_end_y": X[-1, 1],
        "tilt_end_z": X[-1, 2]
    }


def compute_azimuth_features(X, sr=SR):
    "Features derived from 3D azimuth."
    duration = len(X) / sr
    # Kent features for the 3D azimuth
    vmf_azimuth = extract_vmf_features(X, prepend="azimuth_")

    # Angle between the first and last sample
    azimuth3d_angle = get_angle(X[0], X[-1])
    azimuth3d_speed = azimuth3d_angle / duration

    return {
        **vmf_azimuth,
        "azimuth3d_angle": azimuth3d_angle,
        "azimuth3d_speed": azimuth3d_speed,

    }


def compute_gyr_earth_features(X, sr=SR):
    "Features derived from gyroscope data in an earth reference frame."
    duration = len(X) / sr
    azimuth2d_cumulative_change = X[:, 2].sum() / sr
    azimuth2d_cumulative_speed = azimuth2d_cumulative_change / duration
    return {"azimuth2d_cumulative_change": azimuth2d_cumulative_change,
            "azimuth2d_cumulative_speed": azimuth2d_cumulative_speed,
            **compute_basic_features(np.cumsum(X[:, 2] / sr),
                                     columns=["azimuth2d_change"],
                                     compute_quartiles=True,
                                     compute_higher_moments=False)}


def compute_imu_features_basic(imu, sr=SR, col_acc_G=COL_ACC_G,
                               col_gyr=COL_GYR, col_acc_R=COL_ACC_R):
    "Compute all basic imu features."
    duration = len(imu) / sr
    basic_features = compute_basic_features(imu[col_gyr + col_acc_G + col_acc_R],
                                            compute_quartiles=True,
                                            compute_higher_moments=False)

    # basic features on the diff
    imu_grad = gradient_df(imu[col_gyr + col_acc_G + col_acc_R])
    basic_features_grad = compute_basic_features(imu_grad,
                                                 compute_quartiles=True,
                                                 compute_higher_moments=False)

    # basic features on the norm of the gyroscope and accR
    gyr_norm = np.linalg.norm(imu[col_gyr].values, axis=1)
    acc_R_norm = np.linalg.norm(imu[col_acc_R].values, axis=1)
    df_norm = pd.DataFrame(np.c_[gyr_norm, acc_R_norm], columns=[
                           "gyr_norm", "acc_R_norm"])
    basic_features_norm = compute_basic_features(df_norm,
                                                 compute_quartiles=True,
                                                 compute_higher_moments=False)

    # ts features on the gyroscope, acc_G and acc_R and the gyr and accR gradients
    ts_features = compute_ts_features(imu[col_gyr + col_acc_R + col_acc_G])
    # Correlation coefficients
    corr_all = compute_corr(imu[COL_ACC_R + COL_GYR])

    return {
        "duration": duration,
        "duration_log10": np.log10(duration),
        **basic_features,
        **basic_features_grad,
        **basic_features_norm,
        **ts_features,
        **corr_all,
        "gyr_energy": np.sum(gyr_norm),
        "acc_R_energy": np.sum(acc_R_norm)
    }


def compute_imu_features_attitude(imu, sr=SR, col_acc_G=COL_ACC_G,
                                  col_gyr=COL_GYR, col_quat=COL_QUAT):
    "Compute all imu features derived from AHRS filters."
    # Azimuth estimate using computed quaternions
    azimuth3d = get_azimuth_estimate(imu[col_quat].values,
                                     first_sample_azimuth_zero=True)
    # Change of referential using computed quaternions
    gyr_earth = R.from_quat(imu[col_quat].values).apply(imu[col_gyr].values)

    return {
        **compute_tilt_features(imu[col_acc_G].values, sr=sr),
        **compute_azimuth_features(azimuth3d, sr=sr),
        **compute_gyr_earth_features(gyr_earth, sr=sr)
    }


def compute_feature_all(imu, chpt, **kwargs):
    "Compute all features on windows of inertial data."
    X_basic = split_and_map(
        imu,
        chpt[:-1],
        lambda x: compute_imu_features_basic(x, **kwargs),
        n_jobs=None
    )
    X_basic = pd.DataFrame(X_basic)

    X_attitude = split_and_map(imu,
                               chpt[:-1],
                               lambda x: compute_imu_features_attitude(x, **kwargs)
    )
    X_attitude = pd.DataFrame(X_attitude)

    return X_basic.join(X_attitude)
