# @file matrix.py
#
#  MATRIX MANIPULATION FUNCTIONS
#
#  Some useful functions for linear algebra and matrix manipulaiton.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2016
#

import numpy as np
from itertools import product


##
#  Function that transposes a multidimensional matrix to the right.
#
#  @param[in] data: The inpud N-dimensional array.
#
#  @return The transposed data.
#
def ftr(data):

    return fancy_transpose(data)


##
#  Function that transposes a multidimensional matrix to the left.
#
#  @param[in] data: The inpud N-dimensional array.
#
#  @return The transposed data.
#
def ftl(data):

    return fancy_transpose(data, -1)


##
#  Function that transposes a multidimensional matrix.
#
#  @param[in] data: The inpud N-dimensional array.
#  @param[in] roll: The roll direction and amount.
#
#  @return The transposed data.
#
def fancy_transpose(data, roll=1):

    axis_roll = np.roll(np.arange(data.ndim), roll)

    return np.transpose(data, axes=axis_roll)


##
#  This function orthonormalizes the row vectors of the input matrix.
#
#  @param[in] matrix: Input matrix.
#  @param[in] return_opt: Option to return u, e or both.
#
#  @return Lists of orthogonal vectors, u, and/or orthonormal vectors, e.
#
def gram_schmidt(matrix, return_opt='orthonormal'):

    u = []
    e = []

    for vector in matrix:

        if len(u) == 0:
            u_now = vector
        else:
            u_now = vector - sum([project(u_i, vector) for u_i in u])

        u.append(u_now)
        e.append(u_now / np.linalg.norm(u_now, 2))

    u = np.array(u)
    e = np.array(e)

    if return_opt == 'orthonormal':
        return e
    elif return_opt == 'orthogonal':
        return u
    else:
        return u, e


##
#  Function that computes the nuclear norm of the input data.
#
#  @param[in] data: Input data.
#
#  @return Nuclear norm.
#
def nuclear_norm(data):

    # Get SVD of the data.
    u, s, v = np.linalg.svd(data)

    # Return nuclear norm.
    return np.sum(s)


##
#  Function that projects vector v onto vector u.
#
#  @param[in] u: Input vector u.
#  @param[in] v: Input vector v.
#
#  @return Projection.
#
def project(u, v):

    return np.inner(v, u) / np.inner(u, u) * u


##
#  Funciton that produces a 2x2 rotation matrix for the given input angle.
#
#  @param[in] angle: Rotation angle in radians.
#
#  @return Rotation matrix.
#
def rot_matrix(angle):

    return np.around(np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]], dtype='float'), 10)


##
#  Funciton that rotates an input matrix about the input angle.
#
#  @param[in] matrix: Input matrix.
#  @param[in] angle: Rotation angle in radians.
#
#  @return Rotated matrix.
#
def rotate(matrix, angle):

    shape = np.array(matrix.shape)

    if shape[0] != shape[1]:
        raise ValueError('Input matrix must be square.')

    shift = (np.array(shape) - 1) / 2

    index = np.array(list(product(*np.array([np.arange(val) for val in
                     shape])))) - shift

    new_index = np.array(np.dot(index, rot_matrix(angle)), dtype='int') + shift
    new_index[new_index >= shape[0]] -= shape[0]

    return matrix[zip(new_index.T)].reshape(shape.T)
