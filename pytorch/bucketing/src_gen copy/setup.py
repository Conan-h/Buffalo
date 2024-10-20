from setuptools import setup, Extension

__version__ = '0.0.1'

# Describe our module distribution to distutils/setuptools
setup(
    name='src_gen',
    version=__version__,
    author='Your Name',
    author_email='your.email@example.com',
    url='http://example.com',
    description='A module for performing operations on lists',
    long_description='',
    ext_modules=[
        Extension(
            'src_gen',
            ['src_gen.cpp'],
            include_dirs=[
                '/home/cc/.local/lib/python3.6/site-packages/pybind11/include/'
            ],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
            language='c++'
        ),
    ],
    zip_safe=False,
)
