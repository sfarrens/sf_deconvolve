#  @file string.py
#
#  STRING FUNCTIONS
#
#  Basic string manipulation functions.
#
#  @author Samuel Farrens
#  @version 1.0
#  @date 2016
#


##
#  Function extracts the numeric values from a string.
#
#  @param[in] string: Input string.
#  @param[in] format: Output format. [int, float]
#
def extract_num(string, format=None):

    numeric = filter(str.isdigit, string)

    if format is int:
        return int(numeric)

    elif format is float:
        return float(numeric)

    else:
        return numeric
