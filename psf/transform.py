#  @file transform.py
#
#  DATA TRANSFORM ROUTINES
#
#  Functions for transforming
#  data. Based on work by
#  Yinghao Ge.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from scipy.ndimage import gaussian_filter
from itertools import islice, product
from functions.np_adjust import data2np


##
#  Function that reformats a 3D data cube into a 2D data map.
#
#  @param[in] data_cube: 3D Input data cube.
#  @param[in] layout: 2D Layout of data map.
#
#  @return Mapped data.
#
#  @exception: ValueError for invalid layout.
#
def cube2map(data_cube, layout):

    if data_cube.shape[0] != np.prod(layout):
        raise ValueError('The desired layout must match the number of input '
                         'data layers.')

    return np.vstack([np.hstack(data_cube[slice(layout[1] * i, layout[1] *
                      (i + 1))]) for i in xrange(layout[0])])


##
#  Function that reformats a 2D data map into a 3D data cube.
#
#  @param[in] data_map: 2D Input data map.
#  @param[in] layout: 2D Layout of data map.
#
#  @return Mapped data.
#
#  @exception: ValueError for invalid layout.
#
def map2cube(data_map, layout):

    if np.all(np.array(data_map.shape) % np.array(layout)) != 0:
        raise ValueError('The desired layout must be a multiple of the number '
                         'pixels in the data map.')

    d_shape = np.array(data_map.shape) / np.array(layout)

    return np.array([data_map[(slice(i * d_shape[0], (i + 1) * d_shape[0]),
                    slice(j * d_shape[1], (j + 1) * d_shape[1]))] for i in
                    xrange(layout[0]) for j in xrange(layout[1])])


##
#  Function that reformats a data map into a matrix.
#
#  @param[in] data_map: Input data map.
#  @param[in] layout: 2D Layout of image map.
#
#  @return Data matrix.
#
def map2matrix(data_map, layout):

    layout = data2np(layout)

    # Select n objects
    n_obj = np.prod(layout)

    # Get the shape of the galaxy images
    gal_shape = (data2np(data_map.shape) / layout)[0]

    # Stack objects from map
    data_matrix = []

    for i in range(n_obj):
        lower = (gal_shape * (i / layout[1]),
                 gal_shape * (i % layout[1]))
        upper = (gal_shape * (i / layout[1] + 1),
                 gal_shape * (i % layout[1] + 1))
        data_matrix.append((data_map[lower[0]:upper[0],
                            lower[1]:upper[1]]).reshape(gal_shape ** 2))

    return np.array(data_matrix).T


##
#  Function that reformats a data matrix into a map.
#
#  @param[in] data_matrix: Input data matrix.
#  @param[in] map_shape: Shape of the output map.
#
#  @return Data cube.
#
def matrix2map(data_matrix, map_shape):

    map_shape = data2np(map_shape)

    # Get the shape and layout of the galaxy images
    gal_shape = np.sqrt(data_matrix.shape[0])
    layout = np.array(map_shape / np.repeat(gal_shape, 2), dtype='int')

    # Map objects from matrix
    data_map = np.zeros(map_shape)

    temp = data_matrix.reshape(gal_shape, gal_shape, data_matrix.shape[1])

    for i in range(data_matrix.shape[1]):
        lower = (gal_shape * (i / layout[1]),
                 gal_shape * (i % layout[1]))
        upper = (gal_shape * (i / layout[1] + 1),
                 gal_shape * (i % layout[1] + 1))
        data_map[lower[0]:upper[0], lower[1]:upper[1]] = temp[:, :, i]

    return data_map


##
#  Function that reformats a data cube into a matrix.
#
#  @param[in] data_cube: 3D Input data cube.
#
#  @return Data matrix.
#
def cube2matrix(data_cube):

    return data_cube.reshape([data_cube.shape[0]] +
                             [np.prod(data_cube.shape[1:])]).T


##
#  Function that reformats a data matrix into a cube.
#
#  @param[in] data_matrix: Input data matrix.
#  @param[in] im_shape: Shape of stack element images.
#
#  @return Data cube.
#
def matrix2cube(data_matrix, im_shape):

    return data_matrix.T.reshape([data_matrix.shape[1]] + list(im_shape))
