#  @file system.py
#
#  SYSTEM FUNCTIONS
#
#  Basic system functions.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2016
#

import os
import numpy as np


##
#  Function returns a numpy array of files in a directory.
#
#  @param[in] path: Directory path.
#  @param[in] string: File identification string.
#
def get_files(path, string=''):

    return np.array([file for file in os.listdir(path) if string in file])
