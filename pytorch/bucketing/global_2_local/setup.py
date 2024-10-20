from setuptools import setup, Extension

__version__ = '0.0.1'

setup(
    name='find_indices',
    version=__version__,
    author='Your Name',
    author_email='your.email@example.com',
    url='http://example.com',
    description='A module for finding indices in a tensor',
    long_description='',
    ext_modules=[
        Extension(
            'find_indices',
            ['find_indices.cpp'],
            include_dirs=[
                '/path/to/your/python/headers',
                '/home/cc/.local/lib/python3.10/site-packages/pybind11/include/'
            ],
            language='c++',
            extra_compile_args=['-std=c++11', '-fopenmp'],
            extra_link_args=['-fopenmp']
        ),
    ],
    zip_safe=False,
)
