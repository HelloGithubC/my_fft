from __future__ import annotations
import joblib as joblib
from my_fft.lib.fftpower import FFTPower_CPP
from my_fft.lib.mesh import ToMesh
import numpy
import numpy as np
import os as os
from scipy.fft._basic import irfftn
from scipy.fft._basic import rfftn
import sys as sys
__all__ = ['FFTPowerCPP', 'FFTPower_CPP', 'Mesh', 'ToMesh', 'irfftn', 'joblib', 'np', 'os', 'rfftn', 'sys']
class FFTPowerCPP:
    @classmethod
    def load(cls, filename):
        ...
    def __init__(self, Nmesh, BoxSize, shotnoise = 0.0):
        ...
    def run(self, field, kmin, kmax, dk, Nmu = None, k_arrays = None, mode = '1d', field_type = 'complex', right = False, linear = True, done_conj = False, nthreads = 1):
        ...
    def save(self, filename):
        ...
class Mesh:
    @classmethod
    def load(cls, input_dir, mode = 'real'):
        ...
    def __init__(self, Nmesh, BoxSize):
        ...
    def is_structured_array(self, arr):
        """
        
                Test if the input array is a structured array
                by testing for `dtype.names`
                
        """
    def r2c(self, field, compensated = False, nthreads = 1):
        ...
    def run_cic(self, data, position = 'Position', weight = None, field_dtype = numpy.float32, norm = False, nthreads = 1):
        ...
    def save(self, output_dir, mode = 'real'):
        ...
