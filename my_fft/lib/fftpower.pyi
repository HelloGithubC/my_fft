from __future__ import annotations
import numpy
__all__ = ['FFTPower_CPP']
class FFTPower_CPP:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def CountNumber(self, numbers: numpy.ndarray[numpy.int32], k_array: numpy.ndarray[numpy.float64], mu_array: numpy.ndarray[numpy.float64], k_x_array: numpy.ndarray[numpy.float64], k_y_array: numpy.ndarray[numpy.float64], k_z_array: numpy.ndarray[numpy.float64], threads: int = 1) -> None:
        ...
    def Digitize(self, bins: numpy.ndarray[numpy.int32], values: numpy.ndarray[numpy.float64], array: numpy.ndarray[numpy.float64], right: bool = True, linear: bool = True, threads: int = 1) -> None:
        ...
    def DoConj(self, complex_field: numpy.ndarray[numpy.complex64], nthreads: int = 1) -> None:
        ...
    def IsConj(self) -> bool:
        ...
    def RunFromComplex(self, power: numpy.ndarray[numpy.complex128], power_mu: numpy.ndarray[numpy.float64], power_k: numpy.ndarray[numpy.float64], power_modes: numpy.ndarray[numpy.int32], complex_field: numpy.ndarray[numpy.complex64], k_array: numpy.ndarray[numpy.float64], mu_array: numpy.ndarray[numpy.float64], k_min: float, k_max: float, k_x_array: numpy.ndarray[numpy.float64], k_y_array: numpy.ndarray[numpy.float64], k_z_array: numpy.ndarray[numpy.float64], mode: str = '1d', right: bool = False, linear: bool = True, do_conj: bool = False, nthreads: int = 1) -> None:
        ...
    def __init__(self, arg0: numpy.ndarray[numpy.float64]) -> None:
        ...
