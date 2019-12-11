"""Module for tomography operators utilizing the cuFFT library.

This module provides the forward and adjoint tomography operators as Python
context managers. This means that instances should be instantiated using a
with-block. The context managers construct was chosen because context managers
are capable of gracefully handling interruptions (CTRL + C) by running their
__exit__ method.

"""

import cupy as cp

from libtike.cufft.radonusfft import radonusfft


class TomoCuFFT(radonusfft):
    """Base class for tomography solvers using the USFFT method on GPU.

    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.

    Attributes
    ----------
    ntheta : int
        The number of projections.
    n, nz : int
        The pixel width and height of the projection.
    center : float
        The location of the rotation center for all slices.
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
        radonusfft.fwd(
            self,
            res.data.ptr,
            cp.asarray(obj, dtype='complex64').data.ptr,
            cp.asarray(theta, dtype='float32').data.ptr,
        )
        return res

    def adj(self, data, theta):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.nz, self.n, self.n], dtype='complex64')
        radonusfft.adj(
            self,
            res.data.ptr,
            cp.asarray(data, dtype='complex64').data.ptr,
            cp.asarray(theta, dtype='float32').data.ptr,
        )
        return res
