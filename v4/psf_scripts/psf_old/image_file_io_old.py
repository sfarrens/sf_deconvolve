#  @file image_file_io.py
#
#  IMAGE FILE INPUT/OUTPUT ROUTINES
#
#  Functions for reading and
#  writing images. Based on work
#  by Yinghao Ge.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from astropy.io import fits


##
#  Function that reads a FITS image file.
#
#  @param[in] file: Input file name.
#
#  @return FITS image array.
#
def read_fits_image(file):

    return fits.getdata(file)


##
#  Function that reads a FITS data file.
#
#  @param[in] file: Input file name.
#
#  @return FITS data array.
#
def read_fits_data(file):

    return fits.getdata(file)


##
#  Function that generates a data cube from
#  a FITS image and its corresponding data.
#
#  @param[in] fits_image: FITS image array.
#  @param[in] fits_data: FITS data array.
#  @param[in] n_gal: Number of galaxies.
#  @param[in] pixel_rad: Pixel radius.
#
#  @return Data cube.
#
#  @exception ValueError for invalid n_gal.
#  @exception ValueError for invalid pixel_rad.
#
def gen_data_cube(fits_image, fits_data, pixel_rad, n_gal=None, rand=False):

    if not n_gal:
        n_gal = fits_data.size

    if n_gal < 1 or n_gal > fits_data.size:
        raise ValueError('The number of galaxies must be within 1 and the size'
                         'of the fits_data [%d].' % fits_data.size)
    if pixel_rad < 0 or pixel_rad > fits_data.size / 2:
        raise ValueError('The pixel radius must be wihin 0 and half the size'
                         'of the fits_data [%d].' % int(fits_data.size / 2))

    index = np.arange(n_gal)

    if rand:
        index = np.random.permutation(np.arange(fits_data.size))[:n_gal]

    cube = []

    for i in range(n_gal):
        lower = (fits_data[index[i]][0] - pixel_rad,
                 fits_data[index[i]][1] - pixel_rad)
        upper = (fits_data[index[i]][0] + pixel_rad + 1,
                 fits_data[index[i]][1] + pixel_rad + 1)
        cube.append(fits_image[lower[0]:upper[0], lower[1]:upper[1]])

    return np.transpose(cube, axes=[1, 2, 0]), index
