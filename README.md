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
$ pip install ahrs==0.3.1 distinctipy==1.2.2 ruptures==1.1.6 ssqueezepy==0.6.3
```



You can now activate the environment and launch jupyter notebook to run the example notebooks:
```bash
$ conda activate dissect
$ jupyter notebook
```

### Additional requirements for polar plots

If you wish to use the 2D projection of 3D trajectories on the unit sphere, follow the [instructions](https://libgeos.org/usage/install/) for installing `geos`.


Then install cartopy and its requirements:
```bash
$ conda install -c conda-forge geos=3.8.1 proj=7.0.0 cartopy=0.17.0
```

Lastly, follow the instructions for installing [rfayat/angle_visualization](https://github.com/rfayat/angle_visualization).

Troubleshooting instructions for installing cartopy are available in the README of [angle_visualization](https://github.com/rfayat/angle_visualization?tab=readme-ov-file#troubleshooting).

