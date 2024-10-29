#include "fftpower.h"
#include <cstdio>
#include <limits>
#include <omp.h>

inline double my_abs(double x)
{
    return (x < 0)? -x : x;
}

FFTPower::FFTPower(py::array_t<double>& boxSize)
{
    if (boxSize.size() != this->ndim)
    {
        throw std::runtime_error("The length of boxSize must be 3.");
    }
    auto boxSize_it = boxSize.mutable_unchecked<1>();
    this->boxSize.resize(this->ndim);
    for (int i=0; i<this->ndim; i++)
    {
        this->boxSize[i] = boxSize_it(i);
    }
}

void FFTPower::Digitize(py::array_t<int>& bins, py::array_t<double>& values, py::array_t<double>& array, bool right, bool linear, int threads)
{
    py::ssize_t bins_size = bins.shape(0);
    py::ssize_t values_size = values.shape(0);
    if (bins.ndim() != 1 || values.ndim() != 1)
    {
        throw std::runtime_error("The length of bins and values must be 1.");
    }
    if (bins_size != values_size)
    {
        throw std::runtime_error("The length of bins and values must be the same.");
    }

    // auto array_info = array.request();
    // int array_size = array_info.shape[0];
    // double* array_ptr = reinterpret_cast<double*>(array_info.ptr);

    auto bins_it = bins.mutable_unchecked<1>();
    auto values_it = values.unchecked<1>();

    int bins_index = 0;
    bool order = (array.at(0) <= array.at(1))? true : false;
    omp_set_num_threads(threads);
    #pragma omp parallel for private(bins_index)
    for(py::ssize_t i=0; i<values_size; i++)
    {
        bins_index = this->FindBin(values_it(i), array, order, right, linear);
        #pragma omp atomic write
        bins_it(i) = bins_index;
    }
}

void FFTPower::CountNumber(py::array_t<int>& numbers, py::array_t<double>& k_array, py::array_t<double>& mu_array, py::array_t<double>& k_x_array, py::array_t<double>& k_y_array, py::array_t<double>& k_z_array, int threads)
{
    // To Do
    if (numbers.ndim() != 2)
    {
        throw std::runtime_error("The length of numbers must be 2.");
    }
    if (k_array.ndim() != 1 || mu_array.ndim() != 1)
    {
        throw std::runtime_error("The length of k_array and mu_array must be 1.");
    }

    if (k_x_array.ndim() != 1 || k_y_array.ndim() != 1 || k_z_array.ndim() != 1)
    {
        throw std::runtime_error("The length of k_x_array, k_y_array and k_z_array must be 1.");
    }
    
    this->CountNumberCore(numbers, k_array, mu_array, k_x_array, k_y_array, k_z_array,threads);
}

void FFTPower::DoConj(py::array_t<std::complex<float>> &complex_field, int nthreads)
{
    auto com_it = complex_field.mutable_unchecked<3>();

    omp_set_num_threads(nthreads);
    #pragma omp for collapse(3) 
    for (int i=0; i<com_it.shape(0); i++)
    {
        for (int j=0; j<com_it.shape(0); j++)
        {
            for (int k=0; k<com_it.shape(0); k++)
            {
                com_it(i, j, k) *= std::conj(com_it(i, j, k));
            }
        }
    }
}

int FFTPower::FindBin(double value, py::array_t<double>& array, bool order, bool right, bool linear)
{
   int li, ri, mi;
   int array_size = array.shape(0);

   auto array_it = array.unchecked<1>();
   if (order)
   {
        if (value < array_it(0))
            return 0;
        if (value > array_it(array_size-1))
            return array_size;
   }
   else 
   {
        if (value > array_it(0))
            return 0;
        if (value < array_it(array_size-1))
            return array_size;
   }
//    if (value == array[0] && right)
//         return 1;
//     if (value == array[array_size-1] && !right)
//         return array_size-1;

    if (linear)
    {
        double delta = array_it(1) - array_it(0);
        double current_bin_double = (value - array_it(0)) / delta;
        int current_bin_int = static_cast<int>(current_bin_double);
        if (std::abs(current_bin_double - current_bin_int) < 1e-8)
        {
            if (right)
            {
                return current_bin_int;
            }
            else
            {
                return current_bin_int + 1;
            }
        }
        return current_bin_int + 1;
    }
    else
    {
        li = 0;
        ri = array_size - 1;
        if (order)
        {
            while (li < ri && ri - li > 1)
            {
                mi = (li + ri) / 2;
                if (std::abs(value - array_it(mi)) < 1e-8)
                {
                    if (right)
                        return mi;
                    else
                        return mi+1;
                }
                if (value > array_it(mi))
                    li = mi;
                else
                    ri = mi;
            }
            if (li == ri || ri - li == 1)
            {
                if (value > array_it(li))
                    return li+1;
                else
                    return li;
            }
            else
            {
                throw std::runtime_error("The li > ri");
            }
        }
        else 
        {
            while (li < ri && ri - li > 1)
            {
                mi = (li + ri) / 2;
                if (std::abs(value - array_it(mi)) < 1e-8)
                {
                    if (right)
                        return mi;
                    else
                        return mi+1;
                }
                if (value > array_it(mi))
                    ri = mi;
                else
                    li = mi;
            }
            if (li == ri || ri - li == 1)
            {
                if (value < array_it(li))
                    return li+1;
                else
                    return li;
            }
            else
            {
                throw std::runtime_error("The li < ri");
            }
        }
    }
}

void FFTPower::CountNumberCore(py::array_t<int>& numbers, py::array_t<double>& k_array, py::array_t<double>& mu_array, py::array_t<double>& k_x_array, py::array_t<double>& k_y_array, py::array_t<double>& k_z_array, int threads)
{
    py::ssize_t k_bins_number = numbers.shape(0);
    py::ssize_t mu_bins_number = numbers.shape(1);

    // py::ssize_t k_array_size = k_array.shape(0);
    // py::ssize_t mu_array_size = mu_array.shape(0);

    py::ssize_t k_x_array_size = k_x_array.shape(0);
    py::ssize_t k_y_array_size = k_y_array.shape(0);
    py::ssize_t k_z_array_size = k_z_array.shape(0);

    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        py::ssize_t i,j,k, k_index, mu_index;
        double k_x, k_y, k_z, k_temp, mu_temp;
        auto k_x_iterator = k_x_array.unchecked<1>();
        auto k_y_iterator = k_y_array.unchecked<1>();
        auto k_z_iterator = k_z_array.unchecked<1>();
        auto numbers_iterator = numbers.mutable_unchecked<2>();

        // auto k_array_info = k_array.request();
        // int k_array_size = k_array_info.shape[0];
        // double* k_array_ptr = reinterpret_cast<double*>(k_array_info.ptr);
        // auto mu_array_info = mu_array.request();
        // int mu_array_size = mu_array_info.shape[0];
        // double* mu_array_ptr = reinterpret_cast<double*>(mu_array_info.ptr);

        bool order = (k_array.at(0) <= k_array.at(1))? true : false;


        #pragma omp for 
        for(i=0; i<k_x_array_size; i++)
        {
            for(j=0; j<k_y_array_size; j++)
            {
                for(k=0; k<k_z_array_size; k++)
                {
                    k_x = k_x_iterator(i);
                    k_y = k_y_iterator(j);
                    k_z = k_z_iterator(k);
                    k_temp = k_x*k_x + k_y*k_y + k_z * k_z;
                    if (k_temp == 0)
                        continue;
                    mu_temp = k_z / k_temp;

                    k_index = FindBin(k_temp, k_array, order,true) - 1;
                    mu_index = FindBin(mu_temp, mu_array, order,true) - 1;
                    if (k_index >= 0 && k_index < k_bins_number && mu_index >= 0 && mu_index < mu_bins_number)
                    {
                        #pragma omp atomic
                        numbers_iterator(k_index,mu_index) += 1;
                    }
                }
            }
        }
    }
}

void FFTPower::RunFromComplex(py::array_t<std::complex<double>>& power, py::array_t<double>& power_mu, py::array_t<double>& power_k, py::array_t<int>& power_modes, py::array_t<std::complex<float>>& complex_field, py::array_t<double>& k_array, py::array_t<double>& mu_array, double k_min, double k_max, py::array_t<double>& k_x_array, py::array_t<double>& k_y_array, py::array_t<double>& k_z_array, std::string mode, bool right, bool linear, bool do_conj, int nthreads)
{
    if (power.ndim() != 2 || power_mu.ndim() != 2 || power_k.ndim() != 2)
    {
        throw std::runtime_error("The ndim of power, power_k and power_mu must be 2.");
    }
    py::ssize_t k_bins_number = power.shape(0);
    py::ssize_t mu_bins_number = power.shape(1);

    if (complex_field.ndim() != this->ndim)
    {
        throw std::runtime_error("The ndim of complex_field must be 3.");
    }
    if (complex_field.shape(0) != k_x_array.shape(0) || complex_field.shape(1) != k_y_array.shape(0) || complex_field.shape(2) != k_z_array.shape(0))
    {
        throw std::runtime_error("The shape of complex_field must be the same as (k_x_array, k_y_array, k_z_array).");
    }

    if (k_array.ndim() != 1 && mu_array.ndim() != 1)
    {
        throw std::runtime_error("The ndim of k_array and mu_array must be 1.");
    }
    if (k_x_array.ndim() != 1 && k_y_array.ndim() != 1 && k_z_array.ndim() != 1)
    {
        throw std::runtime_error("The ndim of k_x_array, k_y_array and k_z_array must be 1.");
    }

    auto com_it = complex_field.mutable_unchecked<3>();
    // #pragma omp for private(i,j,k)
    // for (i=0; i<com_it.shape(0); i++)
    // {
    //     for (j=0; j<com_it.shape(1); j++)
    //     {
    //         for (k=0; k<com_it.shape(2); k++)
    //         {
    //             com_it(i,j,k) *= std::conj(com_it(i,j,k));
    //         }
    //     }
    // }

    auto power_it = power.mutable_unchecked<2>();
    auto power_k_it = power_k.mutable_unchecked<2>();
    auto power_mu_it = power_mu.mutable_unchecked<2>();
    auto k_x_it = k_x_array.unchecked<1>();
    auto k_y_it = k_y_array.unchecked<1>();
    auto k_z_it = k_z_array.unchecked<1>();
    // auto k_it = k_array.unchecked<1>();
    // auto mu_it = mu_array.unchecked<1>();

    auto power_modes_it = power_modes.mutable_unchecked<2>();

    double k_min_limit = std::sqrt(2.0)/2.0 * k_min; 
    double k_max_limit = k_max;

    omp_set_num_threads(nthreads);
    #pragma omp parallel 
    {
        py::ssize_t k_index = 0;
        py::ssize_t mu_index = 0;
        double k_temp = 0.0;
        std::complex<float> boxSize_factor(1.0f, 0.0f); 

        for (int i=0; i<this->ndim; i++)
        {
            boxSize_factor *= this->boxSize[i];
        }

        bool order = (k_array.at(0) <= k_array.at(1))? true : false;

        int factor = 1;
        double mu_temp = 0.0;
        std::vector<std::vector<std::vector<double>>> power_thread(4);
        for (int i=0; i<4; i++)
        {
            power_thread[i] = std::vector<std::vector<double>>(k_bins_number);
            for (int j=0; j<k_bins_number; j++)
            {
                power_thread[i][j] = std::vector<double>(mu_bins_number, 0.0);
            }
        }
        #pragma omp for collapse(3) 
        for (int i=0; i<k_x_array.shape(0); i++)
        {
            for (int j=0; j<k_y_array.shape(0); j++)
            {
                for (int k=0; k<k_z_array.shape(0); k++)
                {
                    if (do_conj)
                        com_it(i,j,k) *= std::conj(com_it(i,j,k)); 
                    k_temp = std::sqrt(k_x_it(i)*k_x_it(i) + k_y_it(j)*k_y_it(j) + k_z_it(k)*k_z_it(k));

                    if (k_temp == 0 || k_temp < k_min_limit || k_temp > k_max_limit)
                    {
                        continue;
                    }
                    k_index = FindBin(k_temp, k_array, order, right, linear) - 1;
                    if (mode == "2d")
                    {
                        mu_temp = k_z_it(k) / k_temp;
                        if (std::abs(mu_temp - mu_array.at(0)) < 1e-8)
                        {
                            mu_index = 0;
                            factor = 1;
                        }
                        else if (mu_temp >= (mu_array.at(mu_bins_number-1)-1e-8))
                        {
                            mu_index = mu_bins_number - 1;
                            factor = 2;
                        }
                        else
                        {
                            mu_index = FindBin(mu_temp, mu_array, order,right, linear) - 1;
                            factor = 2;
                        }
                    }
                    else
                    {
                        mu_index = 0;
                    }
                    if (k_index >= 0 && k_index < k_bins_number && mu_index >= 0 && mu_index < mu_bins_number)
                    {
                        power_thread[0][k_index][mu_index] += com_it(i,j,k).real() * boxSize_factor.real() * factor;
                        power_thread[1][k_index][mu_index] += k_temp * factor;
                        if (mode == "2d")
                        {
                            power_thread[2][k_index][mu_index] += mu_temp * factor;
                        }
                        power_thread[3][k_index][mu_index] += factor;
                    }
                }
            }
        }

        for (int i=0; i<k_bins_number; i++)
        {
            for (int j=0; j<mu_bins_number; j++)
            {
                #pragma omp critical
                {
                    power_it(i,j).real(power_it(i,j).real() + power_thread[0][i][j]);
                }
                #pragma omp atomic
                power_k_it(i,j) += power_thread[1][i][j];
                if (mode == "2d")
                    #pragma omp atomic
                    power_mu_it(i,j) += power_thread[2][i][j];
                #pragma omp atomic
                power_modes_it(i,j) += static_cast<int>(power_thread[3][i][j]);
            }
        }

        // #pragma omp for collapse(2)
    }

    for (int i=0; i<k_bins_number; i++)
    {
        for (int j=0; j<mu_bins_number; j++)
        {
            if (power_modes_it(i,j) > 0)
            {
                power_it(i,j).real(power_it(i,j).real()/power_modes_it(i,j));
                power_k_it(i,j) /= power_modes_it(i,j);
                if (mode == "2d")
                    power_mu_it(i,j) /= power_modes_it(i,j);
            }
            else 
            {
                continue;
            }
        }
    }
}

PYBIND11_MODULE(fftpower, m) {
    py::class_<FFTPower>(m, "FFTPower_CPP")
        .def(py::init<py::array_t<double>&>())
        .def("Digitize", &FFTPower::Digitize, 
             py::arg("bins"), py::arg("values"), py::arg("array"), 
             py::arg("right") = true, py::arg("linear") = true, 
             py::arg("threads") = 1)
        .def("DoConj", &FFTPower::DoConj, py::arg("complex_field"), py::arg("nthreads") =1)
        .def("IsConj", &FFTPower::IsConj)
        .def("CountNumber", &FFTPower::CountNumber, 
             py::arg("numbers"), py::arg("k_array"), 
             py::arg("mu_array"), py::arg("k_x_array"), 
             py::arg("k_y_array"), py::arg("k_z_array"), 
             py::arg("threads") = 1)
        .def("RunFromComplex", &FFTPower::RunFromComplex, 
             py::arg("power"), py::arg("power_mu"), py::arg("power_k"), 
             py::arg("power_modes"),
             py::arg("complex_field"), py::arg("k_array"), 
             py::arg("mu_array"), py::arg("k_min"), py::arg("k_max"), 
             py::arg("k_x_array"), py::arg("k_y_array"), py::arg("k_z_array"), 
             py::arg("mode") = "1d", py::arg("right") = false, py::arg("linear") = true, py::arg("do_conj") = false,
             py::arg("nthreads") = 1);
}