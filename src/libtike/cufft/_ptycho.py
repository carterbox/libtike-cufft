import cupy as cp
import tike.operators as tike

from libtike.cufft.ptychofft import ptychofft
from libtike.cufft._convolution import Convolution
from libtike.cufft._propagation import Propagation


class Ptycho(tike.Ptycho):
    def __init__(self, detector_shape, probe_shape, nscan, nz, n, ntheta,
                 nmodes=1, **kwargs):  # noqa: D102
        """Please see help(Ptycho) for more info."""
        propagation = Propagation(nmodes * ntheta * nscan, detector_shape,
                                  probe_shape, **kwargs)
        diffraction = Convolution(probe_shape, nscan, nz, n, ntheta, **kwargs)
        tike.Ptycho.__init__(self, detector_shape, probe_shape, nscan, nz, n,
            ntheta=ntheta,
            nmodes=nmodes,
            propagation=propagation,
            diffraction=diffraction,
            **kwargs,
        )
