# libtike-cufft

A [CuPy](https://cupy.chainer.org/) and
[CUDA](https://developer.nvidia.com/cuda-toolkit) FFT based library for
ptychography and tomography operators.

## Installation from source

Setting the following environment variables may be required to help CMake find
your preferred compilers.

```bash
export CUDACXX=path-to-cuda-nvcc
export CXX=path-to-c++-compiler
export CUDAHOSTCXX=path-to-c++-compiler
```

Then, install the package in the normal way


```bash
pip install .
```
