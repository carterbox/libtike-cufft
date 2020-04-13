import cupy as cp
from cupyx.scipy.fftpack import get_fft_plan
import numpy as np
import tike.operators


class Propagation(tike.operators.Propagation):
    """A Fourier-based free-space propagation using CuPy."""
    
    def __init__(self, nwaves, detector_shape, probe_shape, **kwargs):
        super(Propagation, self).__init__(nwaves, detector_shape, probe_shape,
                                          **kwargs)
        self.bwaves = min(nwaves, 512)
        self.far = cp.empty(
            (self.bwaves, detector_shape, detector_shape),
            dtype='complex64',
        )
        self.near = cp.empty(
            (self.bwaves, probe_shape, probe_shape),
            dtype='complex64',
        )
        self.plan = get_fft_plan(self.far, axes=(-2, -1))

    def fwd(self, nearplane, **kwargs):
        shape = nearplane.shape
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        farplane = np.empty(
            (self.nwaves, self.detector_shape, self.detector_shape),
            dtype='complex64')
        nearplane = nearplane.reshape(self.nwaves, self.probe_shape,
                                      self.probe_shape)
        with self.plan:
            for batch in range(0, self.nwaves, self.bwaves):
                stride = min(self.bwaves, self.nwaves - batch)
                self.far[:stride] = 0  # TODO: replace with kernel
                self.near[:stride].set(nearplane[batch:batch + stride])
                self.far[:stride, pad:end, pad:end] = self.near[:stride]
                self.far = cp.fft.fftn(
                    self.far,
                    norm='ortho',
                    axes=(-2, -1),
                )
                farplane[batch:batch + stride] = self.far[:stride].get()
        return farplane.reshape(*shape[:-2], *farplane.shape[-2:])

    def adj(self, farplane, **kwargs):
        shape = farplane.shape
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        nearplane = np.empty(
            (self.nwaves, self.probe_shape, self.probe_shape),
            dtype='complex64',
        )
        farplane = farplane.reshape(self.nwaves, self.detector_shape,
                                    self.detector_shape)
        with self.plan:
            for batch in range(0, self.nwaves, self.bwaves):
                stride = min(self.bwaves, self.nwaves - batch)
                self.far[:stride].set(farplane[batch:batch + stride])
                self.far = cp.fft.ifftn(
                    self.far,
                    norm='ortho',
                    axes=(-2, -1),
                )
                nearplane[batch:batch + stride] \
                    = self.far[:stride, pad:end, pad:end].get()
        return nearplane.reshape(*shape[:-2], *nearplane.shape[-2:])
