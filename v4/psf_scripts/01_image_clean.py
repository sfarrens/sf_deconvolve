#!/Users/sfarrens/Documents/Library/anaconda/bin/python
import numpy as np
import matplotlib.pyplot as plt
from psf import *
from image_clean_funcs import *
from functions.shape import Ellipticity

# Input Path
in_path = '/Users/sfarrens/Data/Work/CEA/Fred/Galaxy/galaxy/'
image_file = in_path + 'image-000-0.fits'
data_file = in_path + 'deep_galaxy_catalog-000.fits'

# Output Path
out_path = '/Users/sfarrens/Documents/Projects/PSF/data/'

# Galaxy Image Radius
radius = 20


# ##########################

def part_01():

    print '1- Reading image data from files.'

    global data_cube, fits_data

    fits_image = image_file_io.read_fits_image(image_file)
    fits_data = image_file_io.read_fits_data(data_file)

    data_cube = image_file_io.gen_data_cube(fits_image, fits_data, radius)

    print '   COMPLETE', data_cube.shape


# ##########################

def part_02():

    global data_cube, wave_cube, clean_ellip

    wave_cube = gen_wave_cube(fits_data, radius)
    data_cube = clean_all(data_cube, wave_cube)
    np.save(out_path + 'deep_galaxy_catalog_clean.npy', data_cube)

    print '   COMPLETE', data_cube.shape


# ##########################

def main():

    part_01()
    part_02()

if __name__ == "__main__":
    main()
