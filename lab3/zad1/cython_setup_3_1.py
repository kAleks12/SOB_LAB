from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    include_dirs=[np.get_include()],
    ext_modules=cythonize("z3_1.pyx"),
    zip_safe=False,
)
