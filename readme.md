PSF Tools
==================

@authors Samuel Farrens

Contents
------------
1. [Introduction](#intro_anchor)
2. [Dependenies](#depend_anchor)
3. [Execution](#execute_anchor)

<a name="intro_anchor"></a>
## Introduction

---

This repository contains Python codes and scripts designed for PSF analysis.

The directory **PSF** contains all of the primary functions and classes used for optimisation and analysis. **PSF_SCRIPTS** contains the user scripts that call these functions and classes. **Functions** contains some additional generic functions and tools.

<a name="depend_anchor"></a>
## Dependencies

---

In order to run the scripts in this library the following packages must be installed:

* **<a href="https://www.python.org/download/releases/2.7/" target="_blank">Python 2.7</a>**
[Tested with v 2.7.11]

* **<a href="http://www.numpy.org/" target="_blank">NumPy</a>**
[Tested with v 1.9.2]

* **<a href="http://www.scipy.org/" target="_blank">SciPy</a>** [Tested with v 0.15.1]

* **<a href="http://matplotlib.org/" target="_blank">Matplotlib</a>** [Tested with v 1.4.3]

* The current implementation of wavelet transformations additionally requires the **mr_transform.cc** C++ script from the Sparse2D library in the **<a href="http://www.cosmostat.org/software/isap/" target="_blank">iSap</a>** package [Tested with v 3.1].

The low-rank approximation analysis can be run purely in Python.

<a name="execute_anchor"></a>
## Execution

---

The primary script is **reconstruction_script.py** which is designed to take an observed (*i.e.* with PSF effects and noise) stack of galaxy images and a known PSF, and attempt to reconstruct the original images. The input format are Numpy binary files (.npy).

The script can be run as follows:

> reconstruction_script -i INPUT_IMAGES.npy -p PSF.npy -o OUTPUT_NAME

Where *INPUT_IMAGES.npy* denotes the Numpy binary file containing the stack of observed galaxy images, *PSF.npy* denotes the PSF corresponding to each galaxy image and *OUTPUT_NAME* specifies the output file name.

### Code Options

* **-h, --help:** show this help message and exit
* **-v, --version:** show program's version number and exit
* **-i INPUT, --input INPUT:** Input noisy data file name. (default: None)
* **-p PSF, --psf PSF:** PSF file name. (default: None)
* **--psf_type {fixed,obj_var,pix_var}:** PSF fortmat type. [fixed, obj_var or pix_var] (default: obj_var)
* **--psf_pcs PSF_PCS:** PSF principal components file name. (default: None)
* **--psf_coef PSF_COEF:** PSF coefficients file name. (default: None)
* **--noise_est NOISE_EST:** Initial noise estimate. (default: None)
* **-o OUTPUT, --output OUTPUT:** Output file name. (default: None)
* **-l LAYOUT LAYOUT, --layout LAYOUT LAYOUT:** Map layout. [Map format only] (default: None)
* **-m {all,wave,lowr,grad}, --mode {all,wave,lowr,grad}:** Option to specify the optimisation mode. [all, wave, lowr or grad] (default: all)
* **--opt_type {condat,fwbw,gfwbw}:** Option to specify the optimisation method to beimplemented. [condat or fwbw] (default: condat)
* **-w WAVELET_LEVELS, --wavelet_levels WAVELET_LEVELS:** Number of wavelet levels. (default: 3)
* **--wavelet_type WAVELET_TYPE:** Wavelet type. (default: 1)
* **--wave_thresh_factor WAVE_TF [WAVE_TF ...]:** Wavelet threshold factor. (default: [3.0, 3.0, 4.0])
* **--lowr_thresh_factor LOWR_TF:** Low rank threshold factor. (default: 1)
* **--lowr_thresh_type LOWR_THRESH_TYPE:** Low rank threshold type. [soft or hard] (default: soft)
* **--n_reweights N_REWEIGHTS:** Number of reweightings. (default: 1)
* **--n_iter N_ITER:** Number of iterations. (default: 150)
* **--relax RELAX:** Relaxation parameter (rho_n). (default: 0.5)
* **--data_format DATA_FORMAT:** Data format. (default: cube)
* **--no_pos:** Option to turn off postivity constraint. (default: True)
* **--no_grad:** Option to turn off gradinet calculation. (default: True)
