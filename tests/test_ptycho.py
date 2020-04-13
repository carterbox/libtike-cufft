import lzma
import os
import pickle
import unittest
import time

import numpy as np
import tike.operators
import matplotlib.pyplot as plt

import libtike.cupy


def compare_result(result):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(result.real)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(result.imag)
    plt.colorbar()


def complex_random(*args):
    return np.random.rand(*args) + 1j * np.random.rand(*args)


def time_an_operator(Operator, method, constructor_args, call_args):
    with Operator(*constructor_args) as op:
        start = time.time()
        result = getattr(op, method)(*call_args)
        stop = time.time()
    name = str(op).split()[0].strip('<;')
    print(f'{name} {method} took {stop-start:0.3f} seconds.')
    return result


class TestConvolution(unittest.TestCase):
    """Test whether libtike operators behave the same as tike operators."""
    def setUp(self):
        """Load a dataset for reconstruction."""
        probe_shape = 61
        nscan = 3119
        ntheta, nz, n = 3, 1021, 1031

        self.nearplane = complex_random(ntheta, nscan, probe_shape,
                                        probe_shape).astype('complex64')
        self.probe = complex_random(ntheta, 1, probe_shape,
                                    probe_shape).astype('complex64')
        self.psi = complex_random(ntheta, nz, n).astype('complex64')
        self.scan = np.random.rand(ntheta, nscan, 2).astype('float32')
        self.scan *= np.array([nz - probe_shape, n - probe_shape])

    def test_fwd(self):
        """Check that the fwd operator is correct."""
        call_args = (self.psi, self.scan, self.probe)
        constructor_args = (self.nearplane.shape[-1], self.scan.shape[1],
                            self.psi.shape[1], self.psi.shape[2],
                            self.psi.shape[0])
        print()
        ref = time_an_operator(tike.operators.Convolution, 'fwd',
                               constructor_args, call_args)
        new = time_an_operator(libtike.cupy.Convolution, 'fwd',
                               constructor_args, call_args)
        compare_result(ref[2, 1, 0, 0])
        compare_result(new[2, 1, 0, 0])
        # plt.show()
        plt.close('all')
        np.testing.assert_allclose(ref, new, rtol=1e-6)

    def test_adj(self):
        """Check that the adj operator is correct."""
        call_args = (self.nearplane, self.scan, self.probe)
        constructor_args = (self.nearplane.shape[-1], self.scan.shape[1],
                            self.psi.shape[1], self.psi.shape[2],
                            self.psi.shape[0])
        print()
        ref = time_an_operator(tike.operators.Convolution, 'adj',
                               constructor_args, call_args)
        new = time_an_operator(libtike.cupy.Convolution, 'adj',
                               constructor_args, call_args)
        compare_result(ref[2])
        compare_result(new[2])
        # plt.show()
        plt.close('all')
        np.testing.assert_allclose(ref, new, rtol=1e-6)


class TestPropagation(unittest.TestCase):
    """Test whether libtike operators behave the same as tike operators."""
    def setUp(self):
        """Load a dataset for reconstruction."""
        probe_shape = 61
        detector_shape = 253
        nscan = 3119
        ntheta, nz, n = 3, 1021, 1031

        self.farplane = complex_random(ntheta, nscan, detector_shape,
                                       detector_shape).astype('complex64')
        self.nearplane = complex_random(ntheta, nscan, probe_shape,
                                        probe_shape).astype('complex64')
        self.psi = complex_random(ntheta, nz, n).astype('complex64')
        self.scan = np.random.rand(ntheta, nscan, 2).astype('float32')
        self.scan *= np.array([nz - probe_shape, n - probe_shape])

    def test_fwd(self):
        """Check that the fwd operator is correct."""
        call_args = (self.nearplane, )
        constructor_args = (self.scan.shape[0] * self.scan.shape[1],
                            self.farplane.shape[-1], self.nearplane.shape[-1])
        print()
        ref = time_an_operator(tike.operators.Propagation, 'fwd',
                               constructor_args, call_args)
        new = time_an_operator(libtike.cupy.Propagation, 'fwd',
                               constructor_args, call_args)
        compare_result(ref[2, 1])
        compare_result(new[2, 1])
        # plt.show()
        plt.close('all')
        np.testing.assert_allclose(ref, new, rtol=1e-6)

    def test_adj(self):
        """Check that the adj operator is correct."""
        call_args = (self.farplane, )
        constructor_args = (self.scan.shape[0] * self.scan.shape[1],
                            self.farplane.shape[-1], self.nearplane.shape[-1])
        print()
        ref = time_an_operator(tike.operators.Propagation, 'adj',
                               constructor_args, call_args)
        new = time_an_operator(libtike.cupy.Propagation, 'adj',
                               constructor_args, call_args)
        compare_result(ref[2, 1])
        compare_result(new[2, 1])
        # plt.show()
        plt.close('all')
        np.testing.assert_allclose(ref, new, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
