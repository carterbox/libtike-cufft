import cupy as cp
import tike.operators

from .convolution import Convolution
from .propagation import Propagation


class Ptycho(tike.operators.Ptycho):
    def __init__(self, detector_shape, probe_shape, nscan, nz, n, **kwargs):
        tike.Ptycho.__init__(self, detector_shape, probe_shape, nscan, nz, n,
            propagation=Propagation,
            diffraction=Convolution,
            **kwargs,
        )
