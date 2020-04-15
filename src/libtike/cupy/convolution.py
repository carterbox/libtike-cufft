from importlib_resources import files
from math import ceil

import cupy as cp
import numpy as np
import tike.operators

_cu_source = files('libtike.cupy').joinpath('convolution.cu').read_text()
_patch_kernel = cp.RawKernel(_cu_source, "patch")


class Convolution(tike.operators.Convolution):
    def __enter__(self):
        limit = 2 * 1024**3  # bytes
        self.bscan = min(
            self.nscan,
            (limit - self.nz * self.n * 8) // (8 + self.probe_shape**2 * 8),
        )
        self.scan = cp.empty((self.bscan, 2), dtype='float32')
        self.obj = cp.empty((self.nz, self.n), dtype='complex64')
        self.near = cp.empty((self.bscan, self.probe_shape, self.probe_shape),
                             dtype='complex64')
        self.BS = (512, 1, 1)
        self.GS = (ceil(self.probe_shape**2 / self.BS[0]), self.bscan, 1)
        return self

    def __exit__(self, type, value, traceback):
        del self.scan
        del self.obj
        del self.near

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
        patches = np.empty(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype=psi.dtype,
        )

        for view in range(self.ntheta):
            self.obj.set(psi[view].astype('complex64'))
            for batch in range(0, self.nscan, self.bscan):
                stride = min(self.bscan, self.nscan - batch)
                self.scan[:stride].set(scan[view, batch:batch +
                                            stride].astype('float32'))
                _patch_kernel(self.GS, self.BS,
                              (self.obj, self.near, self.scan, 1, self.nz,
                               self.n, self.probe_shape, stride, True))
                patches[view, batch:batch + stride] = self.near[:stride].get()

        return self.reshape_patches(patches) * probe

    def adj(self, nearplane, scan, probe):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        probe = self.reshape_probe(probe)
        nearplane = self.reshape_nearplane(nearplane)
        # If nearplane cannot be reshaped into this shape, then there are not
        # enough scan positions to correctly do this operation.
        nearplane = np.conj(probe) * nearplane
        nearplane = nearplane.reshape(self.ntheta, self.nscan, -1,
                                      self.probe_shape, self.probe_shape)
        nearplane = np.sum(nearplane, axis=2)

        obj = np.empty((self.ntheta, self.nz, self.n), dtype=nearplane.dtype)

        for view in range(self.ntheta):
            self.obj[:] = 0
            for batch in range(0, self.nscan, self.bscan):
                stride = min(self.bscan, self.nscan - batch)
                self.scan[:stride].set(scan[view, batch:batch +
                                            stride].astype('float32'))
                self.near[:stride].set(nearplane[view, batch:batch +
                                                 stride].astype('complex64'))
                _patch_kernel(self.GS, self.BS,
                              (self.obj, self.near, self.scan, 1, self.nz,
                               self.n, self.probe_shape, stride, False))
            obj[view] = self.obj.get()

        return obj

    def adj_probe(self, nearplane, scan, psi):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        nearplane = self.reshape_nearplane(nearplane)
        patches = np.empty(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype=psi.dtype,
        )

        for view in range(self.ntheta):
            self.obj.set(psi[view].astype('complex64'))
            for batch in range(0, self.nscan, self.bscan):
                stride = min(self.bscan, self.nscan - batch)
                self.scan[:stride].set(scan[view, batch:batch +
                                            stride].astype('float32'))
                _patch_kernel(self.GS, self.BS,
                              (self.obj, self.near, self.scan, 1, self.nz,
                               self.n, self.probe_shape, stride, True))
                patches[view, batch:batch + stride] = self.near[:stride].get()

        patches = self.reshape_patches(patches)
        return np.conj(patches) * nearplane