# Makefile for building C++ tasks with pybind11

# Compiler and flags
CXX ?= g++ # Specific C++ compiler
BUILD_TYPE ?= Release
ifeq ($(BUILD_TYPE), Release)
    CXXFLAGS = -O2 -std=c++11 -Wall -shared -fPIC $(shell python3 -m pybind11 --includes) -fopenmp
else ifeq ($(BUILD_TYPE), Debug)
    CXXFLAGS = -O0 -std=c++11 -Wall -shared -fPIC $(shell python3 -m pybind11 --includes) -fopenmp
else
    $(error Invalid BUILD_TYPE: $(BUILD_TYPE). Use Release or Debug.)
endif

LDFLAGS = -fopenmp
OUT_SUFFIX = $(shell python3-config --extension-suffix)

# Directories
SRC_DIR = src
LIB_DIR = my_fft/lib

# Targets
.PHONY: all fftpower mesh clean

# Default target
all: fftpower mesh

# Build fftpower
fftpower: $(SRC_DIR)/fftpower.cpp
	$(CXX) $(CXXFLAGS) $< -o $(LIB_DIR)/fftpower${OUT_SUFFIX} $(LDFLAGS)

# Build mesh
mesh: $(SRC_DIR)/mesh.cpp
	$(CXX) $(CXXFLAGS) $< -o $(LIB_DIR)/mesh${OUT_SUFFIX} $(LDFLAGS)

# Clean up
clean:
	rm -f $(LIB_DIR)/*.so