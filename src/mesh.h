#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class ToMesh
{
private:
    static const int ndim = 3; // Only support ndim = 3
    std::vector<double> boxSize;
    std::vector<int> Nmesh;

    // int indexConvert(int* indexs);
    int createIndexsVector(std::vector<std::vector<int>>& indexs);

public:
    ToMesh(){}
    ToMesh(py::array_t<int>& Nmesh, py::array_t<double>& boxSize);

    void SetNmeshBoxSize(py::array_t<int>& Nmesh, py::array_t<double>& boxSize);

    template<typename T_DATA, typename T_MESH>
    void RunCIC(py::array_t<T_DATA>& position, py::array_t<T_DATA>& weight, py::array_t<T_MESH>& mesh, int processors);
    template<typename T_DATA, typename T_MESH>
    void CoreCIC(py::array_t<T_DATA>& position, py::array_t<T_DATA>& weight, py::array_t<T_MESH>& mesh, int processors);


    template<typename T_MESH>
    void DoCompensated(py::array_t<std::complex<T_MESH>> &mesh, 
    py::array_t<std::complex<T_MESH>> &freq_x, py::array_t<std::complex<T_MESH>> &freq_y, py::array_t<std::complex<T_MESH>> &freq_z, 
    int processors);
};