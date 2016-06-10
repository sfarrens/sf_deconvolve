PSF Tools
==================

@authors Samuel Farrens

Contents
------------
1. [Introduction](#intro_anchor)
2. [Dependencies](#depend_anchor)
3. [Execution](#exe_anchor)
 * [Example](#eg_anchor)
 * [Code Options](#opt_anchor)

<a name="intro_anchor"></a>
## Introduction

This repository contains Python codes and scripts designed for PSF analysis.

The directory **PSF** contains all of the primary functions and classes used for optimisation and analysis. **PSF_SCRIPTS** contains the user scripts that call these functions and classes. **Functions** contains some additional generic functions and tools.

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

The primary script is **reconstruction_script.py** which is designed to take an observed (*i.e.* with PSF effects and noise) stack of galaxy images and a known PSF, and attempt to reconstruct the original images. The input format are Numpy binary files (.npy).

The script can be run as follows:

> reconstruction_script -i INPUT_IMAGES.npy -p PSF.npy -o OUTPUT_NAME

Where *INPUT_IMAGES.npy* denotes the Numpy binary file containing the stack of observed galaxy images, *PSF.npy* denotes the PSF corresponding to each galaxy image and *OUTPUT_NAME* specifies the output file name.

<a name="eg_anchor"></a>
### Example

The following example can be run on the sample data provided [here](https://www.dropbox.com/sh/qh6rt76nd5a1ve5/AAD1semtINAwUn6bI5D42Urma?dl=0).

This example takes a sample of 100 galaxy images with PSF effects and added noise and the corresponding PSFs and recovers the original images using low-rank approximation and Condat-Vu optimisation.

> reconstruction_script.py -i example_image_stack.npy -p example_psf.npy -o example_output --mode lowr

The result will be two Numpy binary files called *example_output_primal.npy* and *example_output_dual.npy* corresponding to the primal and dual variables in the splitting algorithm. The reconstructed images will be in the *example_output_primal.npy* file.

<a name="opt_anchor"></a>
### Code Options

* **-h, --help:** Show the help message and exit.

* **-v, --version:** Show the program's version number and exit.

* **-i INPUT, --input INPUT:** Input data file name. File should be a Numpy binary containing a stack of noisy galaxy images with PSF effects. (default: None)

* **-p PSF, --psf PSF:** PSF file name. File should be a Numpy binary containing either: (a) a single PSF (fixed format), (b) a stack of PSFs corresponding to each of the galaxy images (obj_var format), (c) a stack of PSFs corresponding to each of the pixels in each of the galaxy images (pix_var format). (default: None)

* **--psf_type {fixed,obj_var,pix_var}:** PSF format type [fixed, obj_var or pix_var]. This option is used to specify the type of PSF being used. *fixed* corresponds to a single PSF for all galaxy images, *obj_var* corresponds to a unique PSF for each galaxy image and *pix_var* corresponds to a unique PSF for each pixel in each galaxy image. (default: obj_var)

* **--psf_pcs PSF_PCS:** PSF principal components file name. File should be a Numpy binary containing the principal components of a stack of pixel variant PSFs. Only use for *psf_type = pix_var*. (default: None)

* **--psf_coef PSF_COEF:** PSF coefficients file name. File should be a Numpy binary containing the coefficients of a stack of pixel variant PSFs. Only use for *psf_type = pix_var*. (default: None)

* **--noise_est NOISE_EST:** Initial estimate of the noisy in the observed galaxy images. If not specified this quantity is automatically calculated using the median absolute deviation. (default: None)

* **-o OUTPUT, --output OUTPUT:** Output file name. (default: None)

* **-l LAYOUT LAYOUT, --layout LAYOUT LAYOUT:** Image layout for images in *map* format.  Only use for *data_format = map*. (default: None)

* **-m {all,wave,lowr,grad}, --mode {all,wave,lowr,grad}:** Option to specify the optimisation mode [all, wave, lowr or grad]. *all* performs optimisation using both low-rank approximation and wavelets, *wave* using only wavelets, *lowr* uses only low-rank and *grad* uses only gradient descent. (default: all)

* **--opt_type {condat,fwbw,gfwbw}:** Option to specify the optimisation method to be implemented [condat or fwbw]. *condat* implements the Condat-Vu proximal splitting method, *fwbw* implements Forward-Backward splitting with FISTA speed-up and *gfwbw* implements the generalised Forward-Backward splitting method. (default: condat)

* **-w WAVELET_LEVELS, --wavelet_levels WAVELET_LEVELS:** Number of wavelet levels to be used (see [iSap Documentation](http://www.cosmostat.org/wp-content/uploads/2014/12/doc_iSAP.pdf)). (default: 3)

* **--wavelet_type WAVELET_TYPE:** Type of Wavelet to be used (see [iSap Documentation](http://www.cosmostat.org/wp-content/uploads/2014/12/doc_iSAP.pdf)). (default: 1)

* **--wave_thresh_factor WAVE_TF [WAVE_TF ...]:** Wavelet threshold factor. (default: [3.0, 3.0, 4.0])

* **--lowr_thresh_factor LOWR_TF:** Low rank threshold factor. (default: 1)

* **--lowr_thresh_type LOWR_THRESH_TYPE:** Low rank threshold type [soft or hard]. (default: soft)

* **--n_reweights N_REWEIGHTS:** Number of reweightings. (default: 1)

* **--n_iter N_ITER:** Number of iterations. (default: 150)

* **--relax RELAX:** Relaxation parameter (rho_n in Condat-Vu method). (default: 0.5)

* **--data_format DATA_FORMAT:** Data format [map or cube]. This specified if the input data is a stack of galaxy images or a single map of all of the images. (default: cube)

* **--no_pos:** Option to turn off positivity constraint.

* **--no_grad:** Option to turn off gradient calculation.
