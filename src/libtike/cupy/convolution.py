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
        return self

    def __exit__(self, type, value, traceback):
        pass

    def reshape_psi(self, x):
        """Return x reshaped like an object."""
        return x.reshape(self.ntheta, self.nz, self.n)

    def reshape_probe(self, x):
        """Return x reshaped like a probe."""
        x = x.reshape(self.ntheta, -1, self.fly, self.nmode, self.probe_shape,
                      self.probe_shape)
        assert x.shape[1] == 1 or x.shape[1] == self.nscan // self.fly
        return x

    def reshape_nearplane(self, x):
        """Return x reshaped like a nearplane."""
        return x.reshape(self.ntheta, self.nscan // self.fly, self.fly,
                         self.nmode, self.probe_shape, self.probe_shape)

    def reshape_patches(self, x):
        """Return x reshaped like a object patches."""
        return x.reshape(self.ntheta, self.nscan // self.fly, self.fly, 1,
                         self.probe_shape, self.probe_shape)

    def fwd(self, psi, scan, probe):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        psi = self.reshape_psi(psi)
        probe = self.reshape_probe(probe)
        patches = cp.empty(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype=psi.dtype,
        )
        _patch_kernel(
            self.grids,
            self.blocks,
            (psi, patches, scan, self.ntheta, self.nz, self.n,
             self.probe_shape, self.nscan, True),
        )
        return self.reshape_patches(patches) * probe

    def adj(self, nearplane, scan, probe):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        probe = self.reshape_probe(probe)
        nearplane = self.reshape_nearplane(nearplane)
        # If nearplane cannot be reshaped into this shape, then there are not
        # enough scan positions to correctly do this operation.
        nearplane = cp.conj(probe) * nearplane
        nearplane = nearplane.reshape(self.ntheta, self.nscan, -1,
                                      self.probe_shape, self.probe_shape)
        nearplane = cp.sum(nearplane, axis=2)
        obj = cp.zeros((self.ntheta, self.nz, self.n), dtype=nearplane.dtype)
        _patch_kernel(
            self.grids,
            self.blocks,
            (obj, nearplane, scan, self.ntheta, self.nz, self.n,
             self.probe_shape, self.nscan, False),
        )
        return obj

    def adj_probe(self, nearplane, scan, psi):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        nearplane = self.reshape_nearplane(nearplane)
        patches = cp.empty(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype=psi.dtype,
        )
        _patch_kernel(
            self.grids,
            self.blocks,
            (psi, patches, scan, self.ntheta, self.nz, self.n,
             self.probe_shape, self.nscan, True),
        )
        patches = self.reshape_patches(patches)
        return cp.conj(patches) * nearplane
