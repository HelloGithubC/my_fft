# my_fft
A simple python module to do cic and calculate FFTPower P(k,\mu) based on C++ and pybind11

## INSTALL 

### Dependency
CXX compilers: support c++11 and openmp 
others: pybind11 

### Method 
1. setup.py: pip install . (Recommend)

2. Makefile: make && pip install .

3. CMake: mkdir build; cd build; cmake ../ -DCMAKE_BUILD_TYPE="Release"; make; and then do pip install .

### Details 
0. "pip install ." can check if lib files exist. If they exist, will just install the package without compilation.

1. For Makefile: it's neccessary to install pybind11 using pip or conda if you don't want to change CXXFLAGS by yourself.

2. For CMake: you need to change the path of cmake files of pybind11 in CMakeLists.txt by yourself. Or you need to delete the path of them if you install pybind11 by cmake.

### Others 
I provide conan file at conanfile.txt 