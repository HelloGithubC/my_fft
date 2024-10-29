import os
import subprocess
from setuptools import setup, Extension

# Define the library name and paths
lib_dir = 'lib'
suffix = os.popen("python3-config --extension-suffix").read()
fftpower_lib = os.path.join(lib_dir, f'fftpower{suffix}')
mesh_lib = os.path.join(lib_dir, f'mesh{suffix}')

# Function to check if the required shared libraries exist
def check_libraries():
    return os.path.exists(fftpower_lib) and os.path.exists(mesh_lib)

# Function to build the libraries using make
def build_libraries():
    try:
        # Run the make command to build the libraries
        subprocess.check_call(['make'], cwd=os.path.dirname(os.path.abspath(__file__)))
    except subprocess.CalledProcessError as e:
        print("Error during the build process:", e)
        raise

# Check if the libraries exist, and build them if they don't
if not check_libraries():
    print("Libraries not found. Building the libraries...")
    build_libraries()
else:
    print("Libraries found. Skipping build.")

# Setup configuration
setup(
    name='my_fft',
    version='0.1.0',
    description='A Python wrapper for FFT operations using Pybind11',
    author='Your Name',
    author_email='your.email@example.com',
    packages=['my_fft'],
    ext_modules=[
        Extension('my_fft.fftpower', [fftpower_lib]),
        Extension('my_fft.mesh', [mesh_lib]),
    ],
    install_requires=[
        'numpy>=1.16.0',  # Minimum version compatible with Python 3.6
        'scipy>=1.2.0',   # Minimum version compatible with Python 3.6
    ], 
)
