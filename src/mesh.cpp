#include "mesh.h"
#include <cstdio>
#include <cmath>
#include <omp.h>

// int ToMesh::indexConvert(int* indexs)
// {
//     std::vector<int>& Nmesh = this->Nmesh;
//     int ndim = this->ndim;
//     int index = 0;
//     int factor = 1;
//     for (int i=ndim-1; i>=0; i--)
//     {
//         factor = (i<ndim-2)? Nmesh[i]: 1;
//         index += indexs[i]*factor;
//     }
//     return index;
// }

int ToMesh::createIndexsVector(std::vector<std::vector<int>>& indexs)
{
    int ndim = this->ndim;
    int number = 1 << ndim; // 2^ndim
    for (int i = 0; i < number; ++i) {
        std::vector<int> current;
        for (int j = 0; j < ndim; ++j) {
            current.push_back((i >> j) & 1);
        }
        indexs.push_back(current);
    }
    return indexs.size();
}
ToMesh::ToMesh(py::array_t<int> &Nmesh, py::array_t<double> &boxSize)
{
    if (Nmesh.size()!=this->ndim || boxSize.size()!=this->ndim)
    {
        throw std::runtime_error("The length of Nmesh and boxSize must be the same as ndim.");
    }
    this->Nmesh.resize(this->ndim);
    this->boxSize.resize(this->ndim);
    for (int i=0; i<this->ndim; i++)
    {
        this->Nmesh[i] = Nmesh.at(i);
        this->boxSize[i] = boxSize.at(i);
    }
}
void ToMesh::SetNmeshBoxSize(py::array_t<int> &Nmesh, py::array_t<double> &boxSize)
{
    if (Nmesh.size()!=this->ndim || boxSize.size()!=this->ndim)
    {
        throw std::runtime_error("The length of Nmesh and boxSize must be the same as ndim.");
    }
    this->Nmesh.resize(this->ndim);
    this->boxSize.resize(this->ndim);
    for (int i=0; i<this->ndim; i++)
    {
        this->Nmesh[i] = Nmesh.at(i);
        this->boxSize[i] = boxSize.at(i);
    }
}

template<typename T_DATA, typename T_MESH>
void ToMesh::RunCIC(py::array_t<T_DATA>& position, py::array_t<T_DATA>& weight, py::array_t<T_MESH>& mesh, int processors)
{
    int ndim = this->ndim;
    bool need_weight = (weight.size()>0)? true: false;

    if (position.ndim()!=2 || position.shape(1)!=ndim)
    {
        throw std::runtime_error("The dimension of position must be 2 and the second dimension must be the same as ndim.");
    }
    if (need_weight && (weight.ndim()!=1 || weight.size()!=position.shape(0))) 
    {
        throw std::runtime_error("The dimension of weight must be 1 and the shape must be equal as position.shape(0).");
    }
    if (mesh.ndim()!=ndim || mesh.shape(0)!=this->Nmesh[0] || mesh.shape(1)!=this->Nmesh[1] || mesh.shape(2)!=this->Nmesh[2])
    {
        throw std::runtime_error("The dimension of mesh must be 3 and the shape must be equal as this->Nmesh.");
    } 

    this->CoreCIC(position, weight, mesh, processors);
}

template<typename T_DATA, typename T_MESH>
void ToMesh::CoreCIC(py::array_t<T_DATA>& position, py::array_t<T_DATA>& weight, py::array_t<T_MESH>& mesh, int processors)
{
    int i,j,k;
    int ndim = this->ndim;
    bool need_weight = (weight.size()>0)? true: false;

    auto position_it = position.template unchecked<2>();
    auto weight_it = weight.template unchecked<1>();
    auto mesh_it = mesh.template mutable_unchecked<3>();

    omp_set_num_threads(processors);
    #pragma omp parallel 
    {
        std::vector<std::vector<int>> indexs_array;
        createIndexsVector(indexs_array);
        double ratio;
        #pragma omp for private(i,j,k,ratio)
        for (i=0; i<position.shape(0); i++)
        {
            std::vector<int> indexs(ndim, 0);
            std::vector<int> indexs_temp(ndim, 0);
            std::vector<double> indexsDouble(ndim, 0.0);
            double weight = (need_weight)? weight_it(i): 1.0;
            for (j=0; j<ndim; j++)
            {
                indexsDouble[j] = position_it(i,j)/this->boxSize[j]*this->Nmesh[j];
                indexs[j] = static_cast<int>(indexsDouble[j]);
            }
            for (auto indexs_ele: indexs_array)
            {
                ratio = 1.0;
                for (k=0; k<ndim; k++)
                {
                    indexs_temp[k] = indexs[k] + indexs_ele[k];
                    if (indexs_temp[k] >= this->Nmesh[k])
                    {
                        indexs_temp[k] -= this->Nmesh[k];
                    }
                    ratio *= ((indexs_ele[k] == 0)? (indexs[k] + 1 - indexsDouble[k]):(indexsDouble[k] - indexs[k]));
                }
                #pragma omp atomic
                mesh_it(indexs_temp[0], indexs_temp[1], indexs_temp[2]) += weight*ratio;
            }
        }
    }
}

template <typename T_MESH>
void ToMesh::DoCompensated(py::array_t<std::complex<T_MESH>> &mesh, 
    py::array_t<std::complex<T_MESH>> &freq_x, py::array_t<std::complex<T_MESH>> &freq_y, py::array_t<std::complex<T_MESH>> &freq_z, 
    int processors)
{
    int i,j,k;

    auto mesh_it = mesh.template mutable_unchecked<3>();
    auto freq_x_it = freq_x.template unchecked<1>();
    auto freq_y_it = freq_y.template unchecked<1>();
    auto freq_z_it = freq_z.template unchecked<1>();

    omp_set_num_threads(processors);
    std::complex<T_MESH> w[3];
    std::complex<T_MESH> w_temp[3];
    #pragma omp parallel for private(i,j,k,w,w_temp)
    for (i=0; i<mesh.shape(0); i++)
    {
        w[0] = freq_x_it(i);
        w_temp[0] = std::sqrt(static_cast<T_MESH>(1.0) - static_cast<T_MESH>(2.0) / static_cast<T_MESH>(3.0) * std::pow(std::sin(static_cast<T_MESH>(0.5) * w[0]),static_cast<T_MESH>(2.0)));
        for (j=0; j<mesh.shape(1); j++)
        {
            w[1] = freq_y_it(j);
            w_temp[1] = std::sqrt(static_cast<T_MESH>(1.0) - static_cast<T_MESH>(2.0) / static_cast<T_MESH>(3.0) * std::pow(std::sin(static_cast<T_MESH>(0.5) * w[1]),static_cast<T_MESH>(2.0)));
            for (k=0; k<mesh.shape(2); k++)
            {
                w[2] = freq_z_it(k);
                w_temp[2] = std::sqrt(static_cast<T_MESH>(1.0) - static_cast<T_MESH>(2.0) / static_cast<T_MESH>(3.0) * std::pow(std::sin(static_cast<T_MESH>(0.5) * w[2]),static_cast<T_MESH>(2.0)));
                
                mesh_it(i,j,k) = mesh_it(i,j,k) / (w_temp[0] * w_temp[1] * w_temp[2]);
            }
        }
    }
}

template void ToMesh::CoreCIC<double, double>(py::array_t<double>&, py::array_t<double>&, py::array_t<double>&, int);
template void ToMesh::CoreCIC<double, float>(py::array_t<double>&, py::array_t<double>&, py::array_t<float>&, int);
template void ToMesh::CoreCIC<float, double>(py::array_t<float>&, py::array_t<float>&, py::array_t<double>&, int);
template void ToMesh::CoreCIC<float, float>(py::array_t<float>&, py::array_t<float>&, py::array_t<float>&, int);

template void ToMesh::DoCompensated<double>(py::array_t<std::complex<double>>&, py::array_t<std::complex<double>>&, py::array_t<std::complex<double>>&, py::array_t<std::complex<double>>&, int);
template void ToMesh::DoCompensated<float>(py::array_t<std::complex<float>>&, py::array_t<std::complex<float>>&, py::array_t<std::complex<float>>&, py::array_t<std::complex<float>>&, int);


PYBIND11_MODULE(mesh, m)
{
    py::class_<ToMesh>(m, "ToMesh")
        .def(py::init<>())
        .def(py::init<py::array_t<int>&, py::array_t<double>&>())
        .def("SetNmeshBoxSize", &ToMesh::SetNmeshBoxSize,
            py::arg("Nmesh"), py::arg("boxSize"))
        .def("RunCIC", &ToMesh::RunCIC<double, double>, "Run CIC algorithm, T_DATA = double, T_MESH = double.",
            py::arg("position"), py::arg("weight"), py::arg("mesh"), py::arg("processors")=1)
        .def("RunCIC", &ToMesh::RunCIC<double, float>, "Run CIC algorithm, T_DATA = double, T_MESH = float.",
            py::arg("position"), py::arg("weight"), py::arg("mesh"), py::arg("processors")=1)
        .def("RunCIC", &ToMesh::RunCIC<float, double>, "Run CIC algorithm, T_DATA = float, T_MESH = double.",
            py::arg("position"), py::arg("weight"), py::arg("mesh"), py::arg("processors")=1)
        .def("RunCIC", &ToMesh::RunCIC<float, float>, "Run CIC algorithm, T_DATA = float, T_MESH = float.",
            py::arg("position"), py::arg("weight"), py::arg("mesh"), py::arg("processors")=1)
        .def("DoCompensated", &ToMesh::DoCompensated<double>, "Do compensated summation.",
            py::arg("mesh"), py::arg("freq_x"), py::arg("freq_y"), py::arg("freq_z"), py::arg("processors")=1)
        .def("DoCompensated", &ToMesh::DoCompensated<float>, "Do compensated summation.",
            py::arg("mesh"), py::arg("freq_x"), py::arg("freq_y"),py::arg("freq_z"), py::arg("processors")=1);
}
