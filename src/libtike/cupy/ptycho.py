import tike.operators

from .convolution import Convolution
from .propagation import Propagation
from .operator import Operator


class Ptycho(Operator, tike.operators.Ptycho):
    def __init__(self, *args, **kwargs):
        super(Ptycho, self).__init__(
            *args,
            propagation=Propagation,
            diffraction=Convolution,
            **kwargs,
        )
