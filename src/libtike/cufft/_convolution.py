import cupy as cp
import tike.operators
from libtike.cufft.ptychofft import Convolution as _C


class Convolution(_C, tike.operators.Convolution):
    def __init__(self, detector_shape, probe_shape, nscan, nz, n, ntheta=1,
                 **kwargs):  # noqa: D102
        super().__init__(probe_shape, nscan, nz, n, ntheta)

    def fwd(self, psi, scan, **kwargs):
        nearplane = cp.zeros(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype='complex64',
        )
        super().fwd(
            nearplane.data.ptr,
            cp.ascontiguousarray(psi, dtype='complex64').data.ptr,
            cp.ascontiguousarray(scan, dtype='float32').data.ptr,
        )
        return nearplane

    def adj(self, nearplane, scan, **kwargs):
        psi = cp.zeros((self.ntheta, self.nz, self.n), dtype='complex64')
        super().adj(
            cp.ascontiguousarray(nearplane, dtype='complex64').data.ptr,
            psi.data.ptr,
            cp.ascontiguousarray(scan, dtype='float32').data.ptr,
        )
        return psi
