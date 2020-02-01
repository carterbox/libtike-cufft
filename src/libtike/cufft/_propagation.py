
import cupy as cp
import tike.operators
from libtike.cufft.ptychofft import Propagation as _P

class Propagation(_P):

    def __init__(self, nwaves, detector_shape, probe_shape, **kwargs):  # noqa: D102
        """Please see help(Ptycho) for more info."""
        super().__init__(nwaves, detector_shape, probe_shape)

    def fwd(self, nearplane, **kwargs):
        farplane = cp.zeros(
            (self.nwaves, self.detector_shape, self.detector_shape),
            dtype='complex64')
        super().fwd(
            cp.ascontiguousarray(nearplane, dtype='complex64').data.ptr,
            farplane.data.ptr
        )
        return farplane

    def adj(self, farplane, **kwargs):
        """Adjoint Fourier-based free-space propagation operator."""
        nearplane = cp.zeros(
            (self.nwaves, self.probe_shape, self.probe_shape),
            dtype='complex64')
        super().adj(
            nearplane.data.ptr,
            cp.ascontiguousarray(farplane, dtype='complex64').data.ptr
        )
        return nearplane
