import libtike.cufft._propagation as lt
import tike.operators as to
import cupy as cp

import time

import matplotlib.pyplot as plt

def main():

    nwaves = 256
    pw = 256
    dw = 512+16+3

    A = lt.Propagation(nwaves, dw, pw)
    B = to.Propagation(dw, pw, array_module=cp, asnumpy=cp.asnumpy)

    shape = (nwaves, pw, pw)

    nearplane = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)

    amplitude = plt.imread("/home/beams/DCHING/Pictures/images/Cryptomeria_japonica-0256.tif") / 255
    phase = plt.imread("/home/beams/DCHING/Pictures/images/Erdhummel_Bombus_terrestris-0256.tif") / 255
    nearplane[0] = cp.asarray(amplitude + 1j * phase)
    nearplane = cp.ascontiguousarray(nearplane, dtype='complex64')


    start = time.time()
    farplaneB = B.fwd(nearplane)
    stop = time.time()
    print(farplaneB.shape, stop-start)

    start = time.time()
    farplaneA = A.fwd(nearplane)
    stop = time.time()
    print(farplaneA.shape, stop-start)

    return

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(cp.log(cp.abs(farplaneA)).get()[0])
    plt.colorbar()
    plt.title('CUDA')

    plt.subplot(1, 3, 2)
    plt.imshow(cp.log(cp.abs(farplaneB)).get()[0])
    plt.colorbar()
    plt.title('CUPY')

    plt.subplot(1, 3, 3)
    plt.imshow(
        cp.log(
          cp.abs(farplaneA)
          - cp.abs(farplaneB)
        ).get()[0]
    )
    plt.colorbar()
    plt.title('DIFF')
    plt.show()

    # cp.testing.assert_array_equal(farplaneA, farplaneB)


    nearplaneA = A.adj(farplaneB)
    nearplaneB = B.adj(farplaneA)

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(nearplaneA.real.get()[0])
    plt.colorbar()
    plt.title('CUDA')

    plt.subplot(1, 3, 2)
    plt.imshow(nearplaneB.real.get()[0])
    plt.colorbar()
    plt.title('CUPY')

    plt.subplot(1, 3, 3)
    plt.imshow(
        cp.log(
          nearplaneB.real - nearplaneA.real
        ).get()[0]
    )
    plt.colorbar()
    plt.title('DIFF')
    plt.show()

    cp.testing.assert_array_equal(nearplaneA, nearplaneB)

if __name__ == '__main__':
    main()
