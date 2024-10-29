import os
import subprocess
from setuptools import setup, find_packages

# Define the library name and paths
lib_dir = 'myfft/lib'
suffix = os.popen("python3-config --extension-suffix").read().replace("\n", "")
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
    include_package_data=True,
    version='1.0',
    python_requires=">=3.8",
    description='A Python wrapper for FFT operations using Pybind11',
    author='Xiao Liang',
    author_email='xiaoliang5@mail2.sysu.edu.com',
    packages=["my_fft"],
    package_data={
        'my_fft': ['lib/*'],  # 指定 lib 文件夹中的所有文件
    },
    install_requires=[
        'numpy>=1.31.0',  # Minimum version compatible with Python 3.6
        'scipy>=1.20.0',   # Minimum version compatible with Python 3.6
        "joblib",
    ], 
)
