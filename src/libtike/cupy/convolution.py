from importlib_resources import files
from math import ceil

import cupy as cp
import tike.operators

from .operator import Operator

_cu_source = files('libtike.cupy').joinpath('convolution.cu').read_text()
_patch_kernel = cp.RawKernel(_cu_source, "patch")


class Convolution(Operator, tike.operators.Convolution):
    def __enter__(self):
        max_thread = min(self.probe_shape**2,
                         _patch_kernel.attributes['max_threads_per_block'])
        self.blocks = (max_thread, )
        self.grids = (
            -(-self.probe_shape**2 // max_thread),  # ceil division
            self.nscan,
            self.ntheta,
        )
        self.pad = (self.detector_shape - self.probe_shape) // 2
        self.end = self.probe_shape + self.pad
        return self

    def __exit__(self, type, value, traceback):
        pass

    def fwd(self, psi, scan, probe):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        psi = psi.reshape(self.ntheta, self.nz, self.n)
        self._check_shape_probe(probe)
        patches = cp.zeros(
            (self.ntheta, self.nscan // self.fly, self.fly, 1,
             self.detector_shape, self.detector_shape),
            dtype='complex64',
        )
        _patch_kernel(
            self.grids,
            self.blocks,
            (psi, patches, scan, self.ntheta, self.nz, self.n, self.nscan,
             self.probe_shape, self.detector_shape, True),
        )
        patches[..., self.pad:self.end, self.pad:self.end] *= probe
        return patches

    def adj(self, nearplane, scan, probe, obj=None, overwrite=False):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        self._check_shape_nearplane(nearplane)
        self._check_shape_probe(probe)
        if not overwrite:
            nearplane = nearplane.copy()
        nearplane[..., self.pad:self.end, self.pad:self.end] *= cp.conj(probe)
        if obj is None:
            obj = cp.zeros((self.ntheta, self.nz, self.n), dtype='complex64')
        _patch_kernel(
            self.grids,
            self.blocks,
            (obj, nearplane, scan, self.ntheta, self.nz, self.n, self.nscan,
             self.probe_shape, self.detector_shape, False),
        )
        return obj

    def adj_probe(self, nearplane, scan, psi, overwrite=False):
        """Combine probe shaped patches into a probe."""
        self._check_shape_nearplane(nearplane)
        patches = cp.empty(
            (self.ntheta, self.nscan // self.fly, self.fly, 1,
             self.probe_shape, self.probe_shape),
            dtype='complex64',
        )
        _patch_kernel(
            self.grids,
            self.blocks,
            (psi, patches, scan, self.ntheta, self.nz, self.n, self.nscan,
             self.probe_shape, self.probe_shape, True),
        )
        return (nearplane[..., self.pad:self.end, self.pad:self.end] *
                cp.conj(patches))

    def _check_shape_probe(self, x):
        """Check that the probe is correctly shaped."""
        assert type(x) is cp.ndarray, type(x)
        # unique probe for each position
        shape1 = (self.ntheta, self.nscan // self.fly, self.fly, 1,
                  self.probe_shape, self.probe_shape)
        # one probe for all positions
        shape2 = (self.ntheta, 1, 1, 1, self.probe_shape, self.probe_shape)
        if __debug__ and x.shape != shape2 and x.shape != shape1:
            raise ValueError(
                f"probe must have shape {shape1} or {shape2} not {x.shape}")

    def _check_shape_nearplane(self, x):
        """Check that nearplane is correctly shaped."""
        assert type(x) is cp.ndarray, type(x)
        shape1 = (self.ntheta, self.nscan // self.fly, self.fly, 1,
                  self.detector_shape, self.detector_shape)
        if __debug__ and x.shape != shape1:
            raise ValueError(
                f"nearplane must have shape {shape1} not {x.shape}")
