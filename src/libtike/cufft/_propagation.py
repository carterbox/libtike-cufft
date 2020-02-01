import numpy as np
import tike.operators as tike

import libtike.cufft.ptychofft as cpp

class Propagation(cpp.Propagation, tike.Propagation):
    def __init__(self, nwaves, detector_shape, probe_shape,
                 **kwargs):  # noqa: D102
        cpp.Propagation.__init__(self, nwaves, detector_shape, probe_shape)
        tike.Propagation.__init__(self, nwaves, detector_shape, probe_shape,
                                  **kwargs)

    def fwd(self, nearplane, **kwargs):
        shape = nearplane.shape[:-2]
        assert self.nwaves == np.prod(shape)
        nearplane = np.ascontiguousarray(nearplane, dtype='complex64')
        nearplane = nearplane.reshape(-1, self.probe_shape, self.probe_shape)
        farplane = np.zeros(
            (self.nwaves, self.detector_shape, self.detector_shape),
            dtype='complex64')
        super().fwd(
            nearplane.__array_interface__['data'][0],
            farplane.__array_interface__['data'][0],
        )
        return farplane.reshape(*shape, self.detector_shape, self.detector_shape)

    def adj(self, farplane, **kwargs):
        shape = farplane.shape[:-2]
        assert self.nwaves == np.prod(shape)
        farplane = np.ascontiguousarray(farplane, dtype='complex64')
        farplane = farplane.reshape(-1, self.detector_shape, self.detector_shape)
        nearplane = np.zeros(
            (self.nwaves, self.probe_shape, self.probe_shape),
            dtype='complex64')
        super().adj(
            nearplane.__array_interface__['data'][0],
            farplane.__array_interface__['data'][0],
        )
        return nearplane.reshape(*shape, self.probe_shape, self.probe_shape)
