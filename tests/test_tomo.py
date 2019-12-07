import os
import unittest

import dxchange
import numpy as np

import libtike.cufft.tomo as pt

testdir = os.path.dirname(__file__)


class TestTomoSolver(unittest.TestCase):
    """Test the radon usfft operators."""
    def setUp(self):
        """Load the test dataset from the disk."""
        # Model parameters
        self.n = 128  # object size n x,y
        self.nz = 128  # object size in z
        self.ntheta = 128 * 3 // 2  # number of angles (rotations)
        self.center = self.n / 2  # rotation center
        self.theta = np.linspace(0, np.pi,
                                 self.ntheta).astype('float32')  # angles

        self.niter = 128  # tomography iterations
        self.pnz = 128  # number of slice partitions
        # Load object
        beta = dxchange.read_tiff(
            os.path.join(testdir, 'data', 'beta-chip-128.tiff'))
        delta = dxchange.read_tiff(
            os.path.join(testdir, 'data', 'delta-chip-128.tiff'))
        self.u0 = delta + 1j * beta

    def test_adjoint(self):
        """Check that the tomo operators meet adjoint definition."""
        with pt.TomoCuFFT(self.ntheta, self.pnz, self.n, self.center) as slv:
            data = slv.fwd_tomo_batch(self.u0)
            u1 = slv.adj_tomo_batch(data)
            t1 = np.sum(data * np.conj(data))
            t2 = np.sum(self.u0 * np.conj(u1))
            print(f"Adjoint test: {t1.real:06f}{t1.imag:+06f}j "
                  f"=? {t2.real:06f}{t2.imag:+06f}j")
            np.testing.assert_allclose(t1, t2, atol=1e-6)

    def test_simple_interals(self):
        """Check that the fwd tomo operator sums correctly at 0 and PI/2.

        When we project at angles 0 and PI/2, the foward operator should be the
        same as taking the sum over the object array along each axis.
        """
        theta = np.array([0, np.pi/2, np.pi, -np.pi/2], dtype='float32')
        original = np.zeros((1, 128, 128), dtype='complex64')
        original[0, 64, :] += 1
        original[0, :, 64] += 1j
        with pt.TomoCuFFT(theta, len(theta), 1, 128, 128 / 2) as slv:
            data = slv.fwd_tomo_batch(original)
        data1 = np.sum(original, axis=1)
        data2 = np.sum(original, axis=2)
        np.testing.assert_allclose(data[0], data1, atol=1e-6)
        np.testing.assert_allclose(data[1], data2, atol=1e-6)
        np.testing.assert_allclose(data[2], data1, atol=1e-6)
        np.testing.assert_allclose(data[3], data2, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
