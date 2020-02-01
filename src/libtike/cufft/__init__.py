"""Module for operators utilizing the cuFFT library.

This module provides the forward and adjoint operators as Python
context managers. This means that instances should be instantiated using a
with-block. The context managers construct was chosen because context managers
are capable of gracefully handling interruptions (CTRL + C) by running their
__exit__ method.

"""


from pkg_resources import get_distribution, DistributionNotFound

from libtike.cufft._convolution import *
from libtike.cufft._propagation import *
from libtike.cufft._ptycho import *
from libtike.cufft.tomo import *

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
