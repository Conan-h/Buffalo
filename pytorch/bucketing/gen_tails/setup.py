from setuptools import setup, Extension

__version__ = '0.0.1'

setup(
    name='gen_tails',
    version=__version__,
    author='Your Name',
    author_email='your.email@example.com',
    url='http://example.com',
    description='A module for generating tails from lists',
    long_description='',
    ext_modules=[
        Extension(
            'gen_tails',
            ['gen_tails.cpp'],
            include_dirs=[
                '/home/cc/.local/lib/python3.10/site-packages/pybind11/include/'
            ],
            language='c++'
        ),
    ],
    zip_safe=False,
)
