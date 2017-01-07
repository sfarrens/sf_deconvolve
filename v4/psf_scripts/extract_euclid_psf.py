#! /usr/bin/env Python

import numpy as np
import argparse as ap
from os import listdir
from astropy.io import fits
from functions.comp import find_bin
from functions.shape import Ellipticity
from functions.image import downsample, window
from termcolor import colored


##
#  Get global script arguments.
#
def get_opts():

    global opts

    parser = ap.ArgumentParser('SCRIPT OPTIONS:',
                               formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--path', dest='path', default='./',
                        required=False, help='Path to Euclid PSFs.')

    parser.add_argument('-o', '--output', dest='output', default='psf_stack',
                        required=False, help='Output file name.')

    parser.add_argument('-w', '--wavelength', dest='wavelength', default=0.6,
                        type=float, required=False,
                        help='Wavelength of PSF [micron].')

    parser.add_argument('-r', '--radius', dest='radius', default=[20],
                        nargs='+', type=int, required=False,
                        help='Pixel radius.')

    parser.add_argument('-n', '--n_psf', dest='n_psf', type=int,
                        required=False, help='Number of PSFs.')

    parser.add_argument('-s', '--sigma', dest='sigma', default=1000.0,
                        type=float, required=False, help='Ellipticity sigma.')

    parser.add_argument('-f', '--factor', dest='factor', default=1, nargs='+',
                        type=int, required=False, help='Downsampling factor.')

    opts = parser.parse_args()


##
#  Funciton to extract a Euclid PSF of a given wavelength.
#
#  @param[in] file_name: Name of FITS file.
#  @param[in] wavelength: Desired wavelength in microns.
#
#  @return PSF corresponding to input wavelength.
#
def extract_psf(file_name, wavelength, radius, sigma):

    print ' - Extracting PSF from file:', file_name

    # Read the FITS file Header/Data Units HDU.
    hdu = fits.open(file_name, memmap=True)[1:]

    # Set the minimum and maximum wavelengths (wl) available.
    wl_min = hdu[0].header['WLGTH0']
    wl_max = hdu[-1].header['WLGTH0']

    # Set the wavelength bin size.
    bin_size = hdu[1].header['WLGTH0'] - wl_min

    # Check that the requested wavelength is within the limits.
    if wl_min <= wavelength <= wl_max:

        # Find the corresponding bin number.
        hdu_bin = find_bin(wavelength, wl_min, bin_size)

        # Extract PSF.
        psf = downsample(hdu[hdu_bin].data, opts.factor)

        # Calculate PSF centroid.
        cent = np.array(Ellipticity(psf, sigma).centroid, dtype='int')

        # Crop PSF with a 20px radius.
        psf = window(psf, cent, radius)

        # Return the PSF.
        return psf

    else:

        print colored('WARNING', 'yellow') + (': File skipped because the ' +
              'wavelength (' + str(opts.wavelength) + ') is not within the ' +
              'available limits (' + str(wl_min) + ', ' + str(wl_max) + ').')

    # Close the HDU.
    hdu.close()

##
#  Run the script.
#
def run_script():

    # Check path.
    if opts.path[-1] != '/':
        opts.path += '/'

    # Check the pixel radius.
    if not len(opts.radius) in (1, 2):
        raise ValueError('The pixel radius must have only one or two values.')

    print 'Running PSF extraction sctipt.'
    print ' - Path:', opts.path
    print ' - Wavelenth:', opts.wavelength
    print ' - Pixel Radius:', opts.radius
    print ' - Sigma:', opts.sigma
    print ' - Downsampling Factor:', opts.factor

    # Get list of of all the files in the specified path.
    files = listdir(opts.path)[:opts.n_psf]

    # Extract PSF from each file.
    psf_stack = []
    for file in files:
        psf = extract_psf(opts.path + file, opts.wavelength, opts.radius,
                          opts.sigma)
        # Check if the PSF was successfully extracted.
        if not isinstance(psf, type(None)):
            psf_stack.append(psf)
    psf_stack = np.array(psf_stack)

    # Save final PSF stack to a file.
    np.save(opts.output, psf_stack)

    print ' - PSF Shape:', psf_stack.shape
    print ' - Output saved to: ' + opts.output + '.npy'
    print 'Script completed successfully.'

def main():

    get_opts()
    run_script()


if __name__ == "__main__":
    main()
