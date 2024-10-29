cmake_minimum_required(VERSION 3.0)
project(FFTPower)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "")
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED HINTS "/home/xiaoliang/miniconda3/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
# find_package(fmt REQUIRED)

include_directories(${PYBIND11_INCLUDE_DIRS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/my_fft/lib)

# add_executable(main src/fftpower.cpp src/main.cpp)
# pybind11_add_module(fftpower src/fftpower.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -O2")
endif()

# add_executable(test src/test.cpp)
pybind11_add_module(fftpower src/fftpower.cpp)
target_link_libraries(fftpower PUBLIC OpenMP::OpenMP_CXX)

pybind11_add_module(mesh src/mesh.cpp)
target_link_libraries(mesh PUBLIC OpenMP::OpenMP_CXX)
