"""A few helper functions for processing angles.

Parts of the code are from the companion code for FAYAT et al. 2021:
    https://github.com/rfayat/sensors_IMU_head_tilt_rodents
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from ahrs.filters import EKF
import scipy


def get_angle(u, v, degrees=True):
    "Return the angle between two vectors or two sets of vectors."
    dot = np.multiply(u, v).sum(axis=-1)
    u_norm, v_norm = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    dot_normalized = dot / (u_norm * v_norm)
    # Compute the arc-cosine of the normalized dot product
    if degrees:
        return np.degrees(np.arccos(dot_normalized))
    else:
        return np.arccos(dot_normalized)


def compute_quaternion_sensor2earth(gyr, acc, sr=300.):
    """Compute attitude from inertial data, return them with order xyzw.
    
    We use here the best set of hyperparameters for the EKF filter, as
    discussed in FAYAT et al., 2021.
    
    gyr: array (n_samples, 3)
        The gyroscope measurements, in degrees per s.
        
    acc: array (n_samples, 3)
        The accelerometer measurement, in g.
        
    sr: float, default=300Hz
        The sampling rate of the IMU recording.

    """
    f = EKF(gyr=np.radians(gyr), acc=acc,
            frame="ENU",
            frequency=sr,
            var_acc=2e-3,
            var_gyr=.75)
    return f.Q[:, [1, 2, 3, 0]]


def get_gravitational_acc(q_sensor2earth):
    "Estimate gravitational acc in the head ref frame AHRS quaternions."
    r_sensor2earth = R.from_quat(q_sensor2earth)
    r_earth2sensor = r_sensor2earth.inv()
    z_earth = np.array([0., 0., 1.])
    return r_earth2sensor.apply(z_earth)


def get_rot_align_azimuth(u, v):
    """Return a rotation around the z axis aligning the azimuths of u and v.

    u is used as a reference here, i.e. the rotation needs to be applied to v.
    """
    # Compute both azimuths
    u_az = np.arctan2(u[1], u[0])
    v_az = np.arctan2(v[1], v[0])
    # Return the corresponding rotation around the z axis
    angle_rotation = u_az - v_az
    return R.from_euler("z", angle_rotation, degrees=False)


def get_azimuth_estimate(q_sensor2earth, first_sample_azimuth_zero=False):
    "Azimuth estimate from AHRS quaternions."
    r_sensor2earth = R.from_quat(q_sensor2earth)
    x_sensor = np.array([1., 0., 0.])
    azimuth = r_sensor2earth.apply(x_sensor)

    # Align the first sample with the x axis
    if first_sample_azimuth_zero:
        rot = get_rot_align_azimuth(np.array([1., 0., 0.]), azimuth[0])
        azimuth = rot.apply(azimuth)

    return azimuth


def interpolate_saturated(X, threshold=999):
    """Interpolate values of each column > threshold or < -threshold.

    We suppose here that the samples are obtained using a constant sampling
    rate.
    """
    # Interpolate saturated gyroscope values
    is_saturated = (X > threshold) | (X < -threshold)
    X[is_saturated] = np.nan
    X_new = np.zeros_like(X)
    t = np.arange(len(X))  # used as x values for the interpolation
    for i, x in enumerate(X.T):
        is_nan = np.isnan(x)
        interp = scipy.interpolate.interp1d(t[~is_nan],
                                            x[~is_nan],
                                            kind="cubic")
        X_new[:, i] = interp(t)
    return X_new


def get_azimuth_estimate(q_sensor2earth, first_sample_azimuth_zero=False):
    "Azimuth estimate from AHRS quaternions."
    r_sensor2earth = R.from_quat(q_sensor2earth)
    x_sensor = np.array([1., 0., 0.])
    azimuth = r_sensor2earth.apply(x_sensor)
    
    # Align the first sample with the x axis
    if first_sample_azimuth_zero:
        rot = get_rot_align_azimuth(np.array([1., 0., 0.]), azimuth[0])
        azimuth = rot.apply(azimuth)
    
    return azimuth
