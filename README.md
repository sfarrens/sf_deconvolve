PSF Deconvolution
=================

> Author: **Samuel Farrens**

> Year: **2017**

> Version: **3.2**

> Email: **[samuel.farrens@gmail.com](mailto:samuel.farrens@gmail.com)**

Contents
------------
1. [Introduction](#intro_anchor)
1. [Dependencies](#depend_anchor)
1.  [Execution](#exe_anchor)
  1. [Example](#eg_anchor)
  1. [Code Options](#opt_anchor)

<a name="intro_anchor"></a>
## Introduction

This repository contains Python codes and scripts designed for PSF deconvolution and analysis.

The directory ``lib`` contains all of the primary functions and classes used for optimisation and analysis. ``scripts`` contains the user scripts that call these functions and classes. ``functions`` contains some additional generic functions and tools.

<a name="depend_anchor"></a>
## Dependencies

In order to run the scripts in this library the following packages must be installed:

* **[Python 2.7](https://www.python.org/download/releases/2.7/)</a>**
[Tested with v 2.7.11]

* **[Numpy](http://www.numpy.org/)** [Tested with v 1.9.2]

* **[Scipy](http://www.scipy.org/)** [Tested with v 0.15.1]

* **[Matplotlib](http://matplotlib.org/)** [Tested with v 1.4.3]

* The current implementation of wavelet transformations additionally requires the **mr_transform.cc** C++ script from the Sparse2D library in the **[iSap](http://www.cosmostat.org/software/isap/)** package [Tested with v 3.1]. These C++ scripts will be need to be compiled in order to run (see [iSap Documentation](http://www.cosmostat.org/wp-content/uploads/2014/12/doc_iSAP.pdf) for details).

The low-rank approximation analysis can be run purely in Python.

<a name="exe_anchor"></a>
## Execution

The primary script is **deconvolution_script.py** which is designed to take an observed (*i.e.* with PSF effects and noise) stack of galaxy images and a known PSF, and attempt to reconstruct the original images. The input format are Numpy binary files (.npy).

The script can be run as follows:

> deconvolution_script.py -i INPUT_IMAGES.npy -p PSF.npy -o OUTPUT_NAME

Where *INPUT_IMAGES.npy* denotes the Numpy binary file containing the stack of observed galaxy images, *PSF.npy* denotes the PSF corresponding to each galaxy image and *OUTPUT_NAME* specifies the output path and file name.

<a name="eg_anchor"></a>
### Example

The following example can be run on the sample data provided in the ``example`` directory.

This example takes a sample of 100 galaxy images with PSF effects and added noise and the corresponding PSFs and recovers the original images using low-rank approximation and Condat-Vu optimisation.

> deconvolution_script.py -i example_image_stack.npy -p example_psf.npy -o example_output --mode lowr

The result will be two Numpy binary files called *example_output_primal.npy* and *example_output_dual.npy* corresponding to the primal and dual variables in the splitting algorithm. The reconstructed images will be in the *example_output_primal.npy* file.

<a name="opt_anchor"></a>
### Code Options

#### Required Arguments

* **-i INPUT, --input INPUT:** Input data file name. File should be a Numpy binary containing a stack of noisy galaxy images with PSF effects (*i.e.* a 3D array).

* **-p PSF, --psf PSF:** PSF file name. File should be a Numpy binary containing either: (a) a single PSF (*i.e.* a 2D array for *fixed* format) or (b) a stack of PSFs corresponding to each of the galaxy images (*i.e.* a 3D array for *obj_var* format).

#### Optional Arguments

* **-h, --help:** Show the help message and exit.

* **-v, --version:** Show the program's version number and exit.

* **-o, --output:** Output file name. If not specified output files will placed in input file path.

* **--psf_type {fixed,obj_var,pix_var}:** PSF format type [fixed, obj_var or pix_var]. This option is used to specify the type of PSF being used. *fixed* corresponds to a single PSF for all galaxy images, *obj_var* corresponds to a unique PSF for each galaxy image and *pix_var* corresponds to a unique PSF for each pixel in each galaxy image. (default: obj_var)

* **--noise_est:** Initial estimate of the noise standard deviation in the observed galaxy images. If not specified this quantity is automatically calculated using the median absolute deviation.

* **-m, --mode {all,sparse,lowr,grad}:** Option to specify the optimisation mode [all, sparse, lowr or grad]. *all* performs optimisation using both low-rank approximation and sparsity, *sparse* using only sparsity, *lowr* uses only low-rank and *grad* uses only gradient descent. (default: lowr)

* **--opt_type {condat,fwbw,gfwbw}:** Option to specify the optimisation method to be implemented [condat, fwbw or gfwbw]. *condat* implements the Condat-Vu proximal splitting method, *fwbw* implements Forward-Backward splitting with FISTA speed-up and *gfwbw* implements the generalised Forward-Backward splitting method. (default: condat)

* **--wavelet_type:** Type of Wavelet to be used (see [iSap Documentation](http://www.cosmostat.org/wp-content/uploads/2014/12/doc_iSAP.pdf)). (default: 1)

* **--wave_thresh_factor:** Wavelet threshold factor. (default: [3.0, 3.0, 4.0])

* **--lowr_thresh_factor:** Low rank threshold factor. (default: 1)

* **--lowr_type:** Type of low-rank regularisation [standard or ngole]. (default: standard)

* **--lowr_thresh_type:** Low rank threshold type [soft or hard]. (default: hard)

* **--n_reweights:** Number of reweightings. (default: 1)

* **--n_iter:** Number of iterations. (default: 150)

* **--relax:** Relaxation parameter (rho_n in Condat-Vu method). (default: 0.8)

* **--condat_sigma:** Condat proximal dual parameter. (default: 0.5)

* **--condat_tau:** Condat proximal primal parameter. (default: 0.5)

* **--kernel:** Standard deviation of pixels for Gaussian kernel. This option will multiply the deconvolution results by a Gaussian kernel.

* **--cost_window:** Window to measure cost function. (default: 1)

* **--no_pos:** Option to turn off positivity constraint.

* **--no_grad:** Option to turn off gradient calculation.
