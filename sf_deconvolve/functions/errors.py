#  @file errors.py
#
#  ERROR FUNCTIONS
#
#  Custom exceptions.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2015
#

import sys
import os.path
import warnings
from termcolor import colored


##
#  Function that creates custom warning messages.
#
#  @param[in] warn_string: Warning message string.
#  @param[in] log: Logging structure.
#
#  @return Error text string.
#
def warn(warn_string, log=None):

    # Print warning to stdout.
    sys.stderr.write(colored('WARNING', 'yellow') + ': ' + warn_string + '\n')

    # Check if a logging structure is provided.
    if not isinstance(log, type(None)):
        warnings.warn(warn_string)


##
#  Function that catches errors prints them to the terminal and optionally
#  saves them to a log.
#
#  @param[in] exception: Caught exception.
#  @param[in] log: Logging structure.
#
#  @return Error text string.
#
def catch_error(exception, log=None):

    # Print exception to stdout.
    stream_txt = colored('ERROR', 'red') + ': ' + str(exception) + '\n'
    sys.stderr.write(stream_txt)

    # Check if a logging structure is provided.
    if not isinstance(log, type(None)):
        log_txt = 'ERROR: ' + str(exception) + '\n'
        log.exception(log_txt)


##
#  Function that checks if the input file
#  name is valid.
#
#  @param[in] file_name: Input file name.
#
#  @exception IOError for invalid file name.
#
def file_name_error(file_name):

    if file_name == '' or file_name[0][0] == '-':
        raise IOError('Input file name not specified.')

    elif os.path.isfile(file_name) == False:
        raise IOError('Input file name [%s] not found!' % file_name)
