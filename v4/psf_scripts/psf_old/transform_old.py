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
from itertools import product
from functions.np_adjust import data2np


##
#  Function that reformats the data cube into a single image map.
#
#  @param[in] data_cube: Input data cube.
#  @param[in] layout: 2D Layout of image map.
#
#  @return Mapped image.
#
def cube2map(data_cube, layout):

    layout = data2np(layout)

    # Select n objects
    n_obj = np.prod(layout)
    temp = data_cube[:, :, :n_obj]

    # Map objects from cube
    data_map = np.zeros((layout * temp.shape[:2]))

    for i in range(n_obj):
        lower = (temp.shape[0] * (i / layout[1]),
                 temp.shape[0] * (i % layout[1]))
        upper = (temp.shape[0] * (i / layout[1] + 1),
                 temp.shape[0] * (i % layout[1] + 1))
        data_map[lower[0]:upper[0], lower[1]:upper[1]] = temp[:, :, i]

    return data_map


##
#  Function that reformats the data map into
#  a matrix.
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
#  Function that reformats the data matrix into a map.
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
