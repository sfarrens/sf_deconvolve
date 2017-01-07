#  @file plotting.py
#
#  PLOTTING ROUTINES
#
#  Classes for plotting.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import numpy as np
import matplotlib.pyplot as plt
from psf.transform import cube2map


def livePlot(data, data_old, iteration, layout=None, seed=1):

    if isinstance(layout, type(None)):
        layout = (5, 5)

    # np.random.seed(seed)
    # data_map = cube2map(np.random.permutation(data[:np.prod(layout)]),
    #                     layout)
    data_map = cube2map(data[:np.prod(layout)], layout)
    data_map_old = cube2map(data_old[:np.prod(layout)], layout)

    diff = np.abs(data_map - data_map_old)

    vmin = 0.0
    vmax = np.max([np.max(x) for x in (data_map, data_map_old, diff)])

    plt.ion()

    plt.subplot(131)
    plt.imshow(data_map, cmap='gist_stern', interpolation='nearest',
               vmin=vmin, vmax=vmax)
    plt.title('Iteration: ' + str(iteration))
    plt.subplot(132)
    plt.imshow(data_map_old, cmap='gist_stern', interpolation='nearest',
               vmin=vmin, vmax=vmax)
    plt.subplot(133)
    plt.imshow(diff, cmap='gist_stern', interpolation='nearest',
               vmin=vmin, vmax=vmax)
    plt.title('Absolute Difference')
    plt.draw()


def liveCost(cost_list, iteration, total_it):

    plt.ion()
    plt.plot(np.log10(cost_list), 'r-', linewidth=2.0)
    plt.title('Cost Function - Iteration: ' + str(iteration))
    plt.xlabel('Iteration')
    plt.ylabel('$\log_{10}$ Cost')
    plt.ylim(0, max(np.log10(cost_list)))
    if not isinstance(total_it, type(None)):
        plt.xlim(0, total_it)
    plt.draw()


def plotCost(cost_list):

    plt.ioff()
    plt.close()

    file_name = 'cost_function.pdf'

    plt.plot(np.log10(cost_list), 'r-')
    plt.title('Cost Function')
    plt.xlabel('Iteration')
    plt.ylabel('$\log_{10}$ Cost')
    plt.savefig(file_name)
    plt.close()

    print 'Saving cost function data to:', file_name


def plotTest(test_list, tolerance):

    file_name = 'convergence_test.pdf'

    plt.plot(test_list, 'r-')
    plt.plot(np.repeat(tolerance, len(test_list)), 'k--')
    plt.title('Convergence Test')
    plt.xlabel('Iteration')
    plt.ylabel('$\|\|X_a - X_b\|\| / \|\|X_a\|\|$')
    plt.savefig(file_name)
    plt.close()

    print 'Saving test function data to:', file_name
