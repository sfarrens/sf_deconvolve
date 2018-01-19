# -*- coding: utf-8 -*-

"""COST FUNCTIONS

This module contains classes of different cost functions for sf_deconvolve.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np
from modopt.opt.cost import costObj


class PSFUpdateCost(costObj):

    def __init__(self, *args, **kwargs):

        self._calc_cost = self._calc_cost_method
        super(PSFUpdateCost, self).__init__(*args, **kwargs)

    def _calc_cost_method(self, *args, **kwargs):

        a = np.sum([op.cost(self._grad._x) for op in self._operators[:-1]])
        b = self._operators[-1].cost(self._psf_grad._psf)

        return a + b
