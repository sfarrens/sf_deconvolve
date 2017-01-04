#  @file reweight.py
#
#  REWEIGHTING CLASSES
#
#  Classes for reweighting optimisation implementations.
#
#  REFERENCES:
#  1) Candes, Wakin and Boyd, Enhancing Sparsity by Reweighting l1
#  Minimization, 2007, Journal of Fourier Analysis and Applications,
#  14(5):877-905. (CWB2007)
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np


##
#  Candes, Wakin and Boyd reweighting class.
#
class cwbReweight():

    ##
    #  Class initializer.
    #
    #  @param[in] weights: Original input weights.
    #  @param[in] operator: Data operator.
    #  @param[in] thresh_factor: Threshold factor.
    #
    def __init__(self, weights, thresh_factor=1):

        self.weights = weights
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = thresh_factor

    ##
    #  Method that updates the weight values.
    #
    #  Section 4 from CWB2007.
    #
    #  NOTES:
    #  Reweighting implemented as w = w (1 / (1 + |x^w|/(n * sigma))).
    #
    #  @param[in] data: Input data.
    #
    def reweight(self, data):

        self.weights *= (1.0 / (1.0 + np.abs(data) / (self.thresh_factor *
                         self.original_weights)))
