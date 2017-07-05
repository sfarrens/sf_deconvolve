##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 0
version_minor = 0
version_micro = 0

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Project descriptions
description = """
XXX
"""
long_description = """
XXX
"""

# Main setup parameters
NAME = "sf_deconvolve"
ORGANISATION = "CEA"
MAINTAINER = "XXX"
MAINTAINER_EMAIL = "XXX"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "XXX"
EXTRAURL = "XXX"
URL = "https://github.com/sfarrens/sf_deconvolve"
DOWNLOAD_URL = "https://github.com/sfarrens/sf_deconvolve"
LICENSE = "XXX"
CLASSIFIERS = CLASSIFIERS
AUTHOR = "XXX"
AUTHOR_EMAIL = "XXX"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["sf_deconvolve"]
REQUIRES = [
    "numpy>=1.11.0",
    "scipy>=0.18.0",
]
EXTRA_REQUIRES = {}
