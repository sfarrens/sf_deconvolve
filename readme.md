PSF Tools
==================

@authors Samuel Farrens

Contents
------------
1. [Introduction](#intro_anchor)
2. [Dependenies](#depend_anchor)

<a name="intro_anchor"></a>
Introduction
------------
This repository contains Python codes and scripts designed for PSF analysis.

<a name="depend_anchor"></a>
Dependencies
------------
In order to run the scripts in this library the following packages must be installed:

* **<a href="https://www.python.org/download/releases/2.7/" target="_blank">Python 2.7</a>**
[Tested with v 2.7.11]

* **<a href="http://www.numpy.org/" target="_blank">NumPy</a>**
[Tested with v 1.9.2]

* **<a href="http://www.scipy.org/" target="_blank">SciPy</a>** [Tested with v 0.15.1]

* **<a href="http://matplotlib.org/" target="_blank">Matplotlib</a>** [Tested with v 1.4.3]

* The current implementation of wavelet transformations additionally requires the **mr_transform.cc** C++ script from the Sparse2D library in the **<a href="http://www.cosmostat.org/software/isap/" target="_blank">iSap</a>** package [Tested with v 3.1].

The low-rank approximation analysis can be run purely in Python.
