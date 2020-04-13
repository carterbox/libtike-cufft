"""Module for operators utilizing the CuPy library.

This module implements the forward and adjoint operators using CuPy. This
removes the need for interface layers like pybind11 or SWIG because kernel
launches and memory management may by accessed from Python.
"""


from pkg_resources import get_distribution, DistributionNotFound

from .convolution import *
from .propagation import *
from .ptycho import *

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
