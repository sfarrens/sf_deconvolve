#!/Users/sfarrens/Documents/Library/anaconda/bin/python

import numpy as np
import matplotlib.pyplot as plt
from psf import *
from functions.matrix import gram_schmidt
from ngole_py.samafied_utils import *

path = '/Users/sfarrens/Documents/Projects/PSF/pixel_variant/data/'
cube_file = 'deep_galaxy_catalog_clean.npy'
layout = (10, 10)


# Script to find the minimum number of princicpal components needed to
# represent the galaxy images.
# ##########################

def part_01():

    print '- Starting Script.'

    data_cube = np.load(path + cube_file)

    def get_perp(gal):

        x = gal.reshape(np.prod(gal.shape))

        u = gram_schmidt(np.array([i.reshape(x.shape) for i in
                                   proj_mat(gal.shape)]))

        return x - np.dot(u.T, np.dot(u, x))

    perp_cube = np.array([get_perp(gal) for gal in data_cube])

    u, s, v = np.linalg.svd(perp_cube)

    p = 0
    while np.sum(s[:p]) / np.sum(s) < 0.99:
        p += 1

    print '- Must retain at least', p + 6, 'principal components.'

    print '- Finishing Script.'

# ##########################

def main():

    part_01()

if __name__ == "__main__":
    main()
