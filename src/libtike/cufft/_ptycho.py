import cupy as cp
import tike.operators as tike

from libtike.cufft.ptychofft import ptychofft
from libtike.cufft._convolution import Convolution
from libtike.cufft._propagation import Propagation


class Ptycho(tike.Ptycho):
    def __init__(self, detector_shape, probe_shape, nscan, nz, n, **kwargs):
        tike.Ptycho.__init__(self, detector_shape, probe_shape, nscan, nz, n,
            propagation=Propagation,
            diffraction=Convolution,
            **kwargs,
        )
