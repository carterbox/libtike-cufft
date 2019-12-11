"""Module for ptychography operators utilizing the cuFFT library.

This module provides the forward and adjoint ptychography operators as Python
context managers. This means that instances should be instantiated using a
with-block. The context managers construct was chosen because context managers
are capable of gracefully handling interruptions (CTRL + C) by running their
__exit__ method.

"""

import cupy as cp

from libtike.cufft.ptychofft import ptychofft


class PtychoCuFFT(ptychofft):
    """Base class for ptychography solvers using the cuFFT library.

    This class is a context manager which provides the basic operators required
    to implement a ptychography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.

    Attributes
    ----------
    nscan : int
        The number of scan positions at each angular view.
    probe_shape : int
        The pixel width and height of the probe illumination.
    detector_shape, detector_shape : int
        The pixel width and height of the detector.
    ntheta : int
        The number of angular partitions of the data.
    n, nz : int
        The pixel width and height of the reconstructed grid.
    ntheta : int
        The number of angular partitions to process together
        simultaneously.
    """

    array_module = cp
    asnumpy = cp.asnumpy

    def __init__(self, nscan, probe_shape, detector_shape, ntheta, nz, n):
        """Please see help(PtychoCuFFT) for more info."""
        super().__init__(ntheta, nz, n, nscan, detector_shape, probe_shape)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd(self, psi, scan, probe):
        """Ptychography transform (FQ)."""
        farplane = cp.zeros(
            (self.ntheta, self.nscan, self.detector_shape, self.detector_shape),
            dtype='complex64',
        )
        ptychofft.fwd(
            self,
            farplane.data.ptr,
            cp.asarray(psi, dtype='complex64').data.ptr,
            cp.asarray(scan, dtype='float32').data.ptr,
            cp.asarray(probe, dtype='complex64').data.ptr,
        )
        return farplane

    def adj(self, farplane, scan, probe):
        """Adjoint ptychography transform (Q*F*)."""
        psi = cp.zeros((self.ntheta, self.nz, self.n), dtype='complex64')
        ptychofft.adj(
            self,
            psi.data.ptr,
            cp.asarray(farplane, dtype='complex64').data.ptr,
            cp.asarray(scan, dtype='float32').data.ptr,
            cp.asarray(probe, dtype='complex64').data.ptr,
            0,  # compute adjoint operator with respect to object
        )
        return psi

    def adj_probe(self, farplane, scan, psi):
        """Adjoint ptychography probe transform (O*F*), object is fixed."""
        probe = cp.zeros(
            (self.ntheta, self.probe_shape, self.probe_shape),
            dtype='complex64',
        )
        ptychofft.adj(
            self,
            cp.asarray(psi, dtype='complex64').data.ptr,
            cp.asarray(farplane, dtype='complex64').data.ptr,
            cp.asarray(scan, dtype='float32').data.ptr,
            probe.data.ptr,
            1,  # compute adjoint operator with respect to probe
        )
        return probe
