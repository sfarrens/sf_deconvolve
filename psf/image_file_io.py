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
from itertools import izip
from astropy.io import fits
from functions.image import FetchWindows


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

    data = fits.getdata(file)

    return np.array(zip(data.field(0), data.field(1)))


##
#  Function that generates a data cube from a FITS image and its corresponding
#  data.
#
#  @param[in] fits_image: FITS image array.
#  @param[in] fits_data: FITS data array.
#  @param[in] pixel_rad: Pixel radius.
#  @param[in] n_obj: Number of objects.
#  @param[in] rand: Select random objects from FITS data.
#
#  @return Data cube.
#
#  @exception ValueError for invalid n_obj.
#  @exception ValueError for invalid pixel_rad.
#
def gen_data_cube(fits_image, fits_data, pixel_rad, n_obj=None, rand=False):

    if (not isinstance(n_obj, type(None)) and
        (n_obj < 1 or n_obj > len(fits_data))):
        raise ValueError('The number of galaxies must be within 1 and the size'
                         'of the fits_data [%d].' % len(fits_data))

    if pixel_rad < 0 or pixel_rad > fits_data.size / 2:
        raise ValueError('The pixel radius must be wihin 0 and half the size'
                         'of the fits_data [%d].' % int(fits_data.size / 2))

    if rand:
        np.random.shuffle(fits_data)

    windows = FetchWindows(fits_image, pixel_rad)
    windows.get_pixels(fits_data[:n_obj])

    return windows.scan()
