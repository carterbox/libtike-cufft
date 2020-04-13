import cupy as cp
import tike.operators

from .convolution import Convolution
from .propagation import Propagation


class Ptycho(tike.operators.Ptycho):
    def __init__(self, *args, **kwargs):
        super(Ptycho, self).__init__(
            *args,
            propagation=Propagation,
            diffraction=Convolution,
            **kwargs,
        )
