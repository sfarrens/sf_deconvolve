#  @file shape.py
#
#  SHAPE ESTIMATION ROUTINES
#
#  Functions for estimating galaxy shapes.
#  Based on work by Fred Ngole.
#
#  REFERENCES:
#  1) Cropper et al., Defining a Weak Lensing Experiment in Space, 2013, MNRAS,
#  431, 3103C. (C2013)
#  2) Baker and Moallem, Iteratively weighted centroiding for Shack-Hartman
#  wave-front sensors, 2007, Optics Express, 15, 8, 5147. (BM2007)
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np

##
#  Class for calculating galaxy ellipticities from quadrupole moments.
#
class Ellipticity():

    ##
    #  Method that initialises the class.
    #
    def __init__(self, data, sigma, centroid=None, moments=None):

        self.data = data
        self.sigma = sigma
        self.ranges = np.array([np.arange(i) for i in data.shape])

        if not isinstance(moments, type(None)):
            self.moments = np.array(moments).reshape(2, 2)
            self.get_ellipse()
        elif isinstance(centroid, type(None)):
            self.get_centroid()
        else:
            self.centroid = centroid
            self.update_weights()
            self.get_moments()

    ##
    #  Method that updates the current values of x and y.
    #
    #  Equation 1 (Exponent) from BM2007.
    #
    def update_xy(self):

        self.x = np.outer(self.ranges[0] - self.centroid[0],
                          np.ones(self.data.shape[1]))
        self.y = np.outer(np.ones(self.data.shape[0]),
                          self.ranges[1] - self.centroid[1])

    ##
    #  Method that updates the current value of the weights.
    #
    #  Equation 1 from BM2007.
    #
    def update_weights(self):

        self.update_xy()
        self.weights = np.exp(-(self.x ** 2 + self.y ** 2) /
                              (2 * self.sigma ** 2))

    ##
    #  Method that updates the current centroid value.
    #
    #  Equations 2a, 2b, 2c and 3 from BM2007.
    #
    def update_centroid(self):

        # Calculate the position moments.
        iw = np.array([np.sum(self.data * self.weights, axis=i)
                       for i in (1, 0)])
        sw = np.sum(iw, axis=1)
        sxy = np.sum(iw * self.ranges, axis=1)

        # Update the centroid value.
        self.centroid = sxy / sw

    ##
    #  Method that calculates the centroid of the image.
    #
    def get_centroid(self, n_iter=10):

        # Set initial value for the weights.
        self.weights = np.ones(self.data.shape)

        # Iteratively calculate the centroid.
        for i in range(n_iter):

            # Update the centroid value.
            self.update_centroid()

            # Update the weights.
            self.update_weights()

        # Calculate the quadrupole moments.
        self.get_moments()

    ##
    #  Method that calculates the quadrupole moments.
    #
    #  Equation 10 from C2013.
    #
    def get_moments(self):

        # Calculate moments.
        q = np.array([np.sum(self.data * self.weights * xi * xj) for xi in
                      (self.x, self.y) for xj in (self.x, self.y)])

        self.moments = (q / np.sum(self.data * self.weights)).reshape(2, 2)

        # Calculate the ellipticities.
        self.get_ellipse()

    ##
    #  Method that cacluates ellipticities from quadrupole moments.
    #
    #  Equations 11 and 12 from C2013.
    #
    def get_ellipse(self):

        # Calculate R^2 (q00 + q11).
        r2 = self.moments[0, 0] + self.moments[1, 1]

        # Calculate the ellipticities.
        self.e = np.array([(self.moments[0, 0] - self.moments[1, 1]) / r2,
                           (self.moments[0, 1] + self.moments[1, 0]) / r2])
