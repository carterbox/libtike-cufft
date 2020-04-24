import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
from cupyx.scipy.fftpack import get_fft_plan
import tike.operators

from .operator import Operator


class Propagation(Operator, tike.operators.Propagation):
    """A Fourier-based free-space propagation using CuPy."""
    def __enter__(self):
        farplane = cp.empty(
            (self.nwaves, self.detector_shape, self.detector_shape),
            dtype='complex64')
        self.plan = get_fft_plan(farplane, axes=(-2, -1))
        del farplane
        return self

    def __exit__(self, type, value, traceback):
        del self.plan
        pass

    def fwd(self, nearplane, overwrite=False, **kwargs):
        self._check_shape(nearplane)
        if overwrite:
            farplane = nearplane
        else:
            farplane = cp.copy(nearplane)
        with self.plan:
            farplane = fftn(
                farplane,
                norm='ortho',
                axes=(-2, -1),
                overwrite_x=True,
            )
        return farplane

    def adj(self, farplane, overwrite=False, **kwargs):
        self._check_shape(farplane)
        if overwrite:
            nearplane = farplane
        else:
            nearplane = cp.copy(farplane)
        with self.plan:
            nearplane = ifftn(
                nearplane,
                norm='ortho',
                axes=(-2, -1),
                overwrite_x=True,
            )
        return nearplane

    def _check_shape(self, x):
        assert type(x) is cp.ndarray, type(x)
        shape = (self.nwaves, self.detector_shape, self.detector_shape)
        if (__debug__ and x.shape[-2:] != shape[-2:]
                and cp.prod(x.shape[:-2]) != self.nwaves):
            raise ValueError(f'waves must have shape {shape} not {x.shape}.')
