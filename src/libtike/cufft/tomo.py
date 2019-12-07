"""Module for tomography."""

import cupy as cp
import numpy as np

from libtike.cufft.radonusfft import radonusfft


class TomoCuFFT(radonusfft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.

    Attribtues
    ----------
    ntheta : int
        The number of projections.
    n, nz : int
        The pixel width and height of the projection.
    """

    array_module = cp
    asnumpy = cp.asnumpy

    def __init__(self, ntheta, nz, n, center):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(ntheta, nz, n, center)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd(self, obj, theta):
        """Radon transform (R)"""
        res = cp.zeros([self.ntheta, self.nz, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        radonusfft.fwd(self, res.data.ptr, obj.data.ptr, theta.data.ptr)
        return res

    def adj(self, data, theta):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.nz, self.n, self.n], dtype='complex64')
        # C++ wrapper, send pointers to GPU arrays
        radonusfft.adj(self, res.data.ptr, data.data.ptr, theta.data.ptr)
        return res
