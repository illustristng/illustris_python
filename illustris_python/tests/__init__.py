"""Tests.
"""

import os
import sys

BASE_PATH_ILLUSTRIS_1 = "/n/ghernquist/Illustris/Runs/L75n1820FP"

# Add path to directory containing 'illustris_python' module
#    e.g. if this file is in '/n/home00/lkelley/illustris/illustris_python/tests/'
this_path = os.path.realpath(__file__)
ill_py_path = os.path.abspath(os.path.join(this_path, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.append(ill_py_path)
import illustris_python as ill
