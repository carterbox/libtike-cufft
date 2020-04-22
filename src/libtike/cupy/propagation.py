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
        pass

    def fwd(self, nearplane, **kwargs):
        assert type(nearplane) is cp.ndarray, type(nearplane)
        shape = nearplane.shape
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        farplane = cp.zeros(
            (self.nwaves, self.detector_shape, self.detector_shape),
            dtype='complex64')
        nearplane = nearplane.reshape(self.nwaves, self.probe_shape,
                                      self.probe_shape)
        farplane[..., pad:end, pad:end] = nearplane
        with self.plan:
            farplane = fftn(
                farplane,
                norm='ortho',
                axes=(-2, -1),
                overwrite_x=True,
            )
        return farplane.reshape(*shape[:-2], *farplane.shape[-2:])

    def adj(self, farplane, **kwargs):
        assert type(farplane) is cp.ndarray, type(farplane)
        shape = farplane.shape
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        farplane = farplane.reshape(self.nwaves, self.detector_shape,
                                    self.detector_shape)
        with self.plan:
            farplane = ifftn(
                farplane,
                norm='ortho',
                axes=(-2, -1),
                overwrite_x=True,
            )
        nearplane = farplane[..., pad:end, pad:end]
        return nearplane.reshape(*shape[:-2], *nearplane.shape[-2:])
