#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

// 模板函数
template <typename T_DATA, typename T_MESH>
void process_arrays(py::array_t<T_DATA>& data_array, py::array_t<T_MESH>& mesh_array) {
    // 使用 unchecked<2>() 迭代 data_array (假设是 2D 数组)
    auto data_buf = data_array.template unchecked<2>();
    size_t rows_data = data_buf.shape(0);
    size_t cols_data = data_buf.shape(1);

    std::cout << "Data Array:" << std::endl;
    for (size_t i = 0; i < rows_data; ++i) {
        for (size_t j = 0; j < cols_data; ++j) {
            std::cout << data_buf(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // 使用 mutable_unchecked<3>() 迭代 mesh_array (假设是 3D 数组)
    auto mesh_buf = mesh_array.template mutable_unchecked<3>();
    size_t dim1_mesh = mesh_buf.shape(0);
    size_t dim2_mesh = mesh_buf.shape(1);
    size_t dim3_mesh = mesh_buf.shape(2);

    std::cout << "Mesh Array before modification:" << std::endl;
    for (size_t i = 0; i < dim1_mesh; ++i) {
        for (size_t j = 0; j < dim2_mesh; ++j) {
            for (size_t k = 0; k < dim3_mesh; ++k) {
                std::cout << mesh_buf(i, j, k) << " ";
            }
            std::cout << std::endl;
        }
    }

    // 修改 mesh_array 的值
    for (size_t i = 0; i < dim1_mesh; ++i) {
        for (size_t j = 0; j < dim2_mesh; ++j) {
            for (size_t k = 0; k < dim3_mesh; ++k) {
                mesh_buf(i, j, k) += 1; // 增加 1
            }
        }
    }

    std::cout << "Mesh Array after modification:" << std::endl;
    for (size_t i = 0; i < dim1_mesh; ++i) {
        for (size_t j = 0; j < dim2_mesh; ++j) {
            for (size_t k = 0; k < dim3_mesh; ++k) {
                std::cout << mesh_buf(i, j, k) << " ";
            }
            std::cout << std::endl;
        }
    }
}

// 注册模块
PYBIND11_MODULE(example, m) {
    m.def("process_arrays", &process_arrays<double, int>, "Process data and mesh arrays");
}