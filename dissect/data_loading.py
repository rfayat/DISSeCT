"Helpers for loading data."
# Hard-coded column names and imu sampling rate
# Could be provided by a parameter file
COL_ACC = [f"a{ax}" for ax in "xyz"]
COL_GYR = [f"g{ax}" for ax in "xyz"]
COL_MAG = [f"m{ax}" for ax in "xyz"]
COL_QUAT = [f"q{ax}" for ax in "xyzw"]
COL_ACC_G = [f"{c}_G" for c in COL_ACC]  # Gravitationnal component of acc
COL_ACC_R = [f"{c}_R" for c in COL_ACC]  # Residual acc (acc grav. - acc)
SR = 300.  # IMU sampling rate, in Herz
