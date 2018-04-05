#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

release_info = {}
infopath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                           "sf_deconvolve_lib", "info.py"))
with open(infopath) as open_file:
    exec(open_file.read(), release_info)

setup(
    name='sf_deconvolve_lib',
    author='sfarrens',
    author_email='samuel.farrens@cea.fr',
    version=release_info["__version__"],
    url='https://github.com/sfarrens/sf_deconvolve',
    download_url='https://github.com/sfarrens/sf_deconvolve',
    packages=find_packages(),
    install_requires=['modopt>=1.1.4', 'sf_tools>=2.0.0'],
    license='MIT',
    description='Galaxy image deconvolution software.',
    long_description=release_info["__about__"],
)
