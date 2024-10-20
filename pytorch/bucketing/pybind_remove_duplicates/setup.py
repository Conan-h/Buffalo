from setuptools import setup, Extension

__version__ = '0.0.1'

# Describe our module distribution to distutils/setuptools
setup(
    name='remove_duplicates',
    version=__version__,
    author='Your Name',
    author_email='your.email@example.com',
    url='http://example.com',
    description='A module for removing duplicated values from a list',
    long_description='',
    ext_modules=[
        Extension(
            'remove_duplicates',
            ['remove_duplicates.cpp'],
            include_dirs=[
                '/home/cc/.local/lib/python3.10/site-packages/pybind11/include/'
            ],
            language='c++'
        ),
    ],
    zip_safe=False,
)
