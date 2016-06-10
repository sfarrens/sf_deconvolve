#!/Users/sfarrens/Documents/Library/anaconda/bin/python

# This script does the following:
# - Reads in the PSF or its principal components corresponding to a given
#   layout.
# - Estimates the noise.
# - Results are saved to .npy files.

import numpy as np
import argparse as ap
from psf import *
from functions.image import *
from functions.extra_math import mfactor
from progressbar import *

path = '/Users/sfarrens/Documents/Projects/PSF/pixel_variant/data/'

def get_opts():

    global opts

    parser = ap.ArgumentParser('SCRIPT OPTIONS:',
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-n', '--n_obj', dest='n_obj', type=int,
                        required=True, help='Number of objects.')

    parser.add_argument('-s', '--sigma', dest='sigma', type=float,
                        required=False, default=2e-5, help='Noise level.')

    parser.add_argument('-p', '--psf', dest='psf', required=False,
                        help='PSF file.')

    parser.add_argument('-t', '--psf_type', dest='psf_type', required=False,
                        default='variant', help='PSF type. [fixed or variant]')

    parser.add_argument('--full_map_psf', dest='full_map_psf',
                        action='store_true', required=False,
                        help='Option to specify if the PSF is in data map '
                        'format.')

    opts = parser.parse_args()


def get_noise_est():

    print '1- Estimating noise from PSF.'

    if opts.full_map_psf:
        form_tag = 'map_'
    else:
        form_tag = 'cube_'

    if opts.psf_type == 'fixed':
        psf_tag = 'fix'
        psf = np.load(path + opts.psf + '.npy')
        psf = psf_pcs.reshape([1] + list(psf_pcs))

    elif opts.psf_type == 'variant':
        psf_tag = 'var'
        psf_pcs = np.load(path + 'psf_pcs_' + form_tag + str(opts.n_obj) +
                          '.npy')
        psf_coef = np.load(path + 'psf_coef_' + form_tag + str(opts.n_obj) +
                           '.npy')

        if not opts.full_map_psf:
            psf_coef = np.transpose(psf_coef, axes=[1, 0, 2, 3])
            psf_coef = psf_coef.reshape([psf_coef.shape[0]] +
                                        list(mfactor(opts.n_obj) *
                                        np.array(psf_coef.shape[2:])))

    kernel_shape = np.array(psf_pcs[0].shape)
    coef_shape = np.array(psf_coef.shape[-2:])

    print coef_shape

    vec_length = np.prod(kernel_shape)

    pcst = [rot_and_roll(x.T) for x in psf_pcs]
    pcs = [rot_and_roll(x) for x in psf_pcs]
    mask = gen_mask(kernel_shape, coef_shape)

    k_pat = kernel_pattern(kernel_shape, mask)

    k_rolls = roll_sequence(kernel_shape)
    m_rolls = roll_sequence(mask.shape)

    res = []
    i = 0
    val = 0

    widgets = ['Test: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=10000000).start()

    for i in range(np.prod(coef_shape)):
        for k1 in range(psf_pcs.shape[0]):
            for k2 in range(psf_pcs.shape[0]):
                pkt = roll_2d(pcst[k1],
                              roll_rad=k_rolls[k_pat[i]]).reshape(vec_length)
                pk = roll_2d(pcs[k2],
                             roll_rad=k_rolls[k_pat[i]]).reshape(vec_length)
                mi = roll_2d(mask, roll_rad=m_rolls[i])
                dk1 = psf_coef[k1][mi]
                dk2 = psf_coef[k2][mi]
                val += sum(pkt * dk1 * dk2 * pk)
        res.append(val)
        pbar.update(10*i+1)

    pbar.finish()

    res = opts.sigma ** 2 * np.array(res).reshape(coef_shape)

    if not opts.full_map_psf:
        res = transform.map2cube(res, mfactor(opts.n_obj))

    np.save(path + 'psf_noise_est_' + form_tag + str(opts.n_obj)+ '_' +
            psf_tag + '.npy', res)


def main():

    get_opts()
    get_noise_est()


if __name__ == "__main__":
    main()
