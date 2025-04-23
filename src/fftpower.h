#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cmath>
#include <vector>


namespace py = pybind11;

class FFTPower
{
private:
    const static int ndim = 3; // Only support 3-dimension
    std::vector<double> boxSize;
    bool doConj;

    int FindBin(double value, py::array_t<double>& array, bool order=true, bool right=false, bool linear=false); // true order means small to big
    void CountNumberCore(py::array_t<int>& numbers, py::array_t<double>& k_array, py::array_t<double>& mu_array, py::array_t<double>& k_x_array, py::array_t<double>& k_y_array, py::array_t<double>& k_z_array, int threads=1);

public:
    FFTPower(py::array_t<double>& boxSize);
    void Digitize(py::array_t<int>& bins, py::array_t<double>& values, py::array_t<double>& array, bool right = true, bool linear = false, int threads = 1);
    void CountNumber(py::array_t<int>& numbers, py::array_t<double>& k_array, py::array_t<double>& mu_array, py::array_t<double>& k_x_array, py::array_t<double>& k_y_array, py::array_t<double>& k_z_array, int threads=1);
    void RunPS3D(py::array_t<std::complex<float>>& complex_field, int nthreads=1);

    bool IsConj() {return this->doConj;}
    void RunFFTPower(py::array_t<std::complex<double>>& power, py::array_t<double>& power_mu, py::array_t<double>& power_k, py::array_t<int>& power_modes, py::array_t<std::complex<float>>& ps_3d, py::array_t<double>& k_array, py::array_t<double>& mu_array, double k_min, double k_max, py::array_t<double>& k_x_array, py::array_t<double>& k_y_array, py::array_t<double>& k_z_array, std::string mode = "1d", bool right=false, bool linear = false, int nthreads=1);
};