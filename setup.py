from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize(Extension(
    "Ising2Dlib", 
    sources=["Ising2D_wrap.pyx", "Ising2D.cpp"], 
    language="c++", 
    include_dirs=[np.get_include()], 
    library_dirs=[], 
    libraries = ["gsl", "gslcblas"], 
    extra_compile_args=[], 
    extra_link_args=[]
), annotate=True))
