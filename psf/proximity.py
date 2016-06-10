#  @file proximity.py
#
#  PROXIMITY OPERATORS
#
#  Classes of proximity operators for optimization.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
from noise import *
from transform import *
from optimisation import *


##
#  Function that retains only the positive elemnts of the input data.
#
#  @param[in] data: Input data.
#
#  @return Positive components of data.
#
def positivity_operator(data):

        return data * (data > 0)


##
#  Positivity proximity operator class.
#
class Positive():

    ##
    #  Class initializer.
    #
    def __init__(self):
        pass

    ##
    #  Class operator.
    #
    #  @param[in] data: Input data.
    #
    #  @return Array of all positive elements from input data.
    #
    def op(self, data, **kwargs):

        return positivity_operator(data)


##
#  Threshold proximity operator class.
#
class Threshold():

    ##
    #  Class initializer.
    #
    #  @param[in] weights: Input weights.
    #  @param[in] positivity: Option to impose positivity constraint.
    #
    def __init__(self, weights, positivity=False):

        self.update_weights(weights)
        self.positivity = positivity

    ##
    #  Method to update current values of the weights
    #
    #  @param[in] weights: Input weights.
    #
    def update_weights(self, weights):

        self.weights = weights

    ##
    #  Class operator.
    #
    #  @param[in] data: Input data.
    #  @param[in] extra_factor: Additional multiplication factor.
    #
    #  @return Thresholded data.
    #
    def op(self, data, extra_factor=1.0):

        threshold = self.weights * extra_factor

        new_data = denoise(data, threshold, 'soft')

        if self.positivity:
            new_data = positivity_operator(data)

        return new_data


##
#  Low rank matrix proximity operator class.
#
class LowRankMatrix():

    ##
    #  Class initializer.
    #
    #  @param[in] thresh: Threshold value.
    #  @param[in] data_format: Input data format. (map or cube)
    #  @param[in] treshold_type: Threshold type. (hard or soft)
    #  @param[in] layout: Data layout.
    #  @param[in] positivity: Option to impose positivity constraint.
    #
    def __init__(self, thresh, data_format='cube', threshold_type='soft',
                 layout=None, positivity=False, operator=None):

        self.thresh = thresh
        self.data_format = data_format
        self.threshold_type = threshold_type
        self.layout = layout
        self.postivity = positivity

        self.operator = operator

    ##
    #  Class operator.
    #
    #  @param[in] data: Input data.
    #  @param[in] extra_factor: Additional multiplication factor.
    #
    #  @return Low-rank thresholded data.
    #
    def op(self, data, extra_factor=1.0):

        # Update threshold with extra factor.
        threshold = self.thresh * extra_factor

        # Subtract mean from data.
        data_mean = np.mean(data)
        data -= data_mean

        # SVD threshold data.
        if self.data_format == 'map':

            if isinstance(self.layout, type(None)):
                raise ValueError('Must specify layout in map mode.')

            data_matrix = svd_threshold(map2matrix(data, self.layout),
                                        threshold,
                                        threshold_type=self.threshold_type)

            new_data = matrix2map(data_matrix, data.shape)

        elif self.data_format == 'cube':

            # data_matrix = svd_thresh(cube2matrix(data), threshold,
            #                          threshold_type=self.threshold_type)

            data_matrix = svd_thresh_coef(data, self.operator,
                                          threshold,
                                          threshold_type=self.threshold_type)

            new_data = matrix2cube(data_matrix, data.shape[1:])

        else:

            raise ValueError('Invalide data format: ' + self.data_format)

        # Add mean back to updated data.
        new_data += data_mean

        # Additional built-in positivity constraint.
        if self.postivity:
            new_data = positivity_operator(new_data)

        # Return updated data.
        return new_data


##
#  Combined proximity operator class.
#
class ProximityCombo():

    ##
    #  Class initializer.
    #
    #  @param[in] operators: List of initialised proximity operator classes.
    #
    def __init__(self, operators):

        self.operators = operators

    ##
    #  Class operator.
    #
    #  @param[in] data: Input data.
    #
    #  @return List of operator outputs.
    #
    def op(self, data, extra_factor=1.0):

        res = np.empty(len(self.operators), dtype=np.ndarray)

        for i in range(len(self.operators)):
            res[i] = self.operators[i].op(data[i], extra_factor=extra_factor)

        return res


##
#  Sub-iteration proximity operator class.
#
class SubIter():

    ##
    #  Class initializer.
    #
    def __init__(self, data_shape, operator, weights=None, u_init=None):

        self.operator = operator

        if not isinstance(weights, type(None)):
            self.weights = weights

        if isinstance(u_init, type(None)):
            self.u = np.ones(data_shape)

        self.opt = ForwardBackward(self.u, self.operator,
                                   Threshold(self.weights), auto_iterate=False,
                                   indent_level=2)

    ##
    #  Method to update current values of the weights.
    #
    #  @param[in] weights: Input weights.
    #
    def update_weights(self, weights):

        self.weights = weights

    ##
    #  Method to update current value of u.
    #
    def update_u(self):

        self.opt.iterate(100)
        self.u = self.opt.x_final

    ##
    #  Class operator.
    #
    #  @param[in] data: Input data.
    #
    #  @return Updated data.
    #
    def op(self, data):

        self.update_u()

        new_data = data - self.operator.adj_op(self.u)

        return new_data
