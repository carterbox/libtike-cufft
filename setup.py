from setuptools import setup, find_namespace_packages

setup(
    name='libtike-cupy',
    author='Viktor Nikitin, Daniel Ching',
    version='0.9.1',
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src', include=['libtike.*']),
    include_package_data=True,
    entry_points={
        'tike.PtychoBackend': [
            'cupy = libtike.cupy:Ptycho',
        ],
        'tike.Propagation': [
            'cupy = libtike.cupy:Propagation',
        ],
        'tike.Convolution': [
            'cupy = libtike.cupy:Convolution',
        ],
    },
    setup_requires=[
        'setuptools',
    ],
    install_requires=[
        'cupy',
        'importlib_resources',
        'numpy',
        'tike',
    ],
)
