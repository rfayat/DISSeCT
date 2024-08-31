# DISSeCT
Decomposition of Inertial Sequences by Segmentation and Clustering in Tandem.

Companion code for Fayat et al. 2024.


## Installation

### Basic installation
Create a virtual environment and install the requirements using conda:
```bash
$ conda env create -f dissect_env.yml
```

Alternatively you can create the environment as follows:
```bash
$ conda create -c conda-forge --name dissect python=3.7.6 cython=0.29.17 ipython=7.13.0 ipykernel=5.1.4 jupyter=1.0.0 matplotlib=3.1.3 notebook=6.0.3 numpy=1.21.5 pandas=1.3.5 scikit-learn=0.23.2 scipy=1.7.3 seaborn=0.12.2 umap-learn=0.5.2
$ conda activate dissect
$ pip install ahrs==0.3.0 distinctipy==1.2.2 ruptures==1.1.6 ssqueezepy==0.6.3
```



You can now activate the environment and launch jupyter notebook to run the example notebooks:
```bash
$ conda activate dissect
$ jupyter notebook
```
### Additional requirement for fast attitude estimate
The pure-python implementation of the Extended Kalman Filter for attitude estimate from inertial data provided in the [ahrs toolbox](https://ahrs.readthedocs.io/en/latest/) is not designed for efficiency and is therefore very slow. To circumvent this issue, I reimplemented some AHRS filters in a compilable version with Cython, using [Mayitzin/ahrs](https://github.com/Mayitzin/ahrs) as Python wrappers, yielding much better computation time while being easy to integrate in computation pipelines written in Python.

After activating your virtual environment, you can install this Cython version of AHRS filters by running the installation instructions from [rfayat/AHRS_cython](https://github.com/rfayat/AHRS_cython).

In case you did not manage to complete this step (e.g. compilation issue), the pipeline will fall back on the pure-Python implementation of the Extended Kalman Filter but will be much slower to run.


### Additional requirements for polar plots

If you wish to use the 2D projection of 3D trajectories on the unit sphere, follow the [instructions](https://libgeos.org/usage/install/) for installing `geos`.


Then install cartopy and its requirements:
```bash
$ conda install -c conda-forge geos=3.8.1 proj=7.0.0 cartopy=0.17.0
```

Lastly, follow the instructions for installing [rfayat/angle_visualization](https://github.com/rfayat/angle_visualization).

Troubleshooting instructions for installing cartopy are available in the README of [angle_visualization](https://github.com/rfayat/angle_visualization?tab=readme-ov-file#troubleshooting).

