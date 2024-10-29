## my_fft
A simple python module to do cic and calculate FFTPower P(k) and P(k,\mu) based on C++ and pybind11. Now support Doing cic 3D mesh and calculating 1D and 2D PS(Pk and Pkmu) of simulation box.

## Features 
1. Easy to use. Without some complex objects. Do cic, do fft and compensating and calculate P(k) and P(k,\mu) just need input simple ndarray data. It means you can insert them to your codes easily.

2. Based on C++ and support openmp. Fast and fully use multiple cores with saving memory.

## INSTALL 

### Dependency
CXX compilers: support c++11 and openmp 
others: pybind11>=1.29.0. Recommend to install it by pip or conda, especially you want to install my_fft wint Method 1.

### Method 
1. setup.py: pip install . (Recommend)

2. Makefile: make && pip install .

3. CMake: mkdir build; cd build; cmake ../ -DCMAKE_BUILD_TYPE="Release"; make. And then go back the root direction to do pip install .

### Details 
0. "pip install ." can check if lib files exist. If they exist, will just install the package without compilation.

1. For Makefile: it's neccessary to install pybind11 using pip or conda if you don't want to change CXXFLAGS by yourself.

2. For CMake: you need to change the path of cmake files of pybind11 in CMakeLists.txt by yourself. Or you need to delete the path of them if you install pybind11 by cmake.

### Others 
I provide conan file at conanfile.txt 