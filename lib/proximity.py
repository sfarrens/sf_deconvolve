# -*- coding: utf-8 -*-

"""PROXIMITY OPERATORS

This module contains classes of proximity operators for optimisation

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from modopt.opt.proximity import ProximityParent


def set_less_than(data, value):

    return (data >= value) * data + (data < value) * value


class ProxPSF(ProximityParent):

    def __init__(self, psf0):

        self.op = lambda x: set_less_than(x, psf0)
        self.cost = lambda x: 0.0
