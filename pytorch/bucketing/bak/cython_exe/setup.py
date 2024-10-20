from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

# setup(ext_modules = cythonize("remove_values.pyx"))

ext_modules=[
    Extension("module1",
        sources=["remove_values.pyx"],
        language='c++',
    ),
]

setup(
    name = "My Project",
    ext_modules = cythonize(ext_modules),
)






