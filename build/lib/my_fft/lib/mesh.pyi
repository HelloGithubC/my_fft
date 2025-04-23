from __future__ import annotations
import numpy
import typing
__all__ = ['ToMesh']
class ToMesh:
    @typing.overload
    def DoCompensated(self, mesh: numpy.ndarray[numpy.complex128], freq_x: numpy.ndarray[numpy.complex128], freq_y: numpy.ndarray[numpy.complex128], freq_z: numpy.ndarray[numpy.complex128], processors: int = 1) -> None:
        """
        Do compensated summation.
        """
    @typing.overload
    def DoCompensated(self, mesh: numpy.ndarray[numpy.complex64], freq_x: numpy.ndarray[numpy.complex64], freq_y: numpy.ndarray[numpy.complex64], freq_z: numpy.ndarray[numpy.complex64], processors: int = 1) -> None:
        """
        Do compensated summation.
        """
    @typing.overload
    def RunCIC(self, position: numpy.ndarray[numpy.float64], weight: numpy.ndarray[numpy.float64], mesh: numpy.ndarray[numpy.float64], processors: int = 1) -> None:
        """
        Run CIC algorithm, T_DATA = double, T_MESH = double.
        """
    @typing.overload
    def RunCIC(self, position: numpy.ndarray[numpy.float64], weight: numpy.ndarray[numpy.float64], mesh: numpy.ndarray[numpy.float32], processors: int = 1) -> None:
        """
        Run CIC algorithm, T_DATA = double, T_MESH = float.
        """
    @typing.overload
    def RunCIC(self, position: numpy.ndarray[numpy.float32], weight: numpy.ndarray[numpy.float32], mesh: numpy.ndarray[numpy.float64], processors: int = 1) -> None:
        """
        Run CIC algorithm, T_DATA = float, T_MESH = double.
        """
    @typing.overload
    def RunCIC(self, position: numpy.ndarray[numpy.float32], weight: numpy.ndarray[numpy.float32], mesh: numpy.ndarray[numpy.float32], processors: int = 1) -> None:
        """
        Run CIC algorithm, T_DATA = float, T_MESH = float.
        """
    def SetNmeshBoxSize(self, Nmesh: numpy.ndarray[numpy.int32], boxSize: numpy.ndarray[numpy.float64]) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32], arg1: numpy.ndarray[numpy.float64]) -> None:
        ...
