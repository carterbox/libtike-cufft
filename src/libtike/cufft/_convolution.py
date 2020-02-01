import numpy as np
import tike.operators as tike

import libtike.cufft.ptychofft as cpp


class Convolution(cpp.Convolution, tike.Convolution):
    def __init__(self, probe_shape, nscan, nz, n, ntheta, **kwargs):
        cpp.Convolution.__init__(self, probe_shape, nscan, nz, n, ntheta)
        tike.Convolution.__init__(self, probe_shape, nscan, nz, n, ntheta, **kwargs)

    def fwd(self, psi, scan, **kwargs):
        nearplane = np.zeros(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype='complex64',
        )
        psi = np.ascontiguousarray(psi, dtype='complex64')
        scan = np.ascontiguousarray(scan, dtype='float32')
        super().fwd(
            nearplane.__array_interface__['data'][0],
            psi.__array_interface__['data'][0],
            scan.__array_interface__['data'][0],
        )
        return nearplane

    def adj(self, nearplane, scan, **kwargs):
        psi = np.zeros((self.ntheta, self.nz, self.n), dtype='complex64')
        nearplane = np.ascontiguousarray(nearplane, dtype='complex64')
        scan = np.ascontiguousarray(scan, dtype='float32')
        super().adj(
            nearplane.__array_interface__['data'][0],
            psi.__array_interface__['data'][0],
            scan.__array_interface__['data'][0],
        )
        return psi
