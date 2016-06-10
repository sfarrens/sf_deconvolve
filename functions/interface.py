#  @file interface.py
#
#  INTERFACE FUNCTIONS
#
#  Functions to add a bit
#  of spice to life...
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import sys


##
#  Function that prints a horizontal line
#  to the terminal.
#
def h_line():

    print '==========================================================='


##
#  Function that prints a progress bar to
#  the terminal.
#
#  @param[in] i: Position in code.
#  E.G. Loop element.
#  @param[in] size: Total number of positions.
#
def progress_bar(i, size):

    sys.stdout.write('\r')
    x = float(i) / float(size) * 100.0
    sys.stdout.write('[%-50s] %7.3f%%' % ('=' * int(x * 0.5), x))
    sys.stdout.flush()
