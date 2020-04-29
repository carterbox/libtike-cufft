# libtike-cupy

A [CuPy](https://cupy.chainer.org/) based library for ptychography and
tomography operators.

## Installation from source

Install the package in the normal way

```bash
pip install .
```

## Using with Tike

To enable these operators, set the following environment variables:

```bash
export TIKE_PTYCHO_BACKEND=cupy
```

These operators replace the default ones in Tike using Python entry points.
