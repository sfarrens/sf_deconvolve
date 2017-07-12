#  @file log.py
#
#  LOG FUNCTIONS
#
#  Logging functions.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2016
#

import sys
import logging


##
#  Set up a basic log.
#
#  @param[in] filename: Log file name.
#
def set_up_log(filename):

    # Add file extension.
    filename += '.log'

    print 'Preparing log file:', filename

    # Capture warnings.
    logging.captureWarnings(True)

    # Set output format.
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')

    # Create file handler.
    fh = logging.FileHandler(filename=filename, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Create log.
    log = logging.getLogger('log')
    log.setLevel(logging.DEBUG)
    log.addHandler(fh)

    # Send test message.
    log.info('The log file has been set-up.')

    return log


##
#  Set up a basic log.
#
#  @param[in] log: Log instance.
#
def close_log(log):

    # Remove all handlers from log.
    [log.removeHandler(handler) for handler in log.handlers]
