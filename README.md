# hothothot

The purpose of this application is to use CUDA cores of an Nvidia GPU to measure heat
diffusion in an object across time slices.

## Structure

### Overview

Nvidia GPUs provide use with a large number of CUDA cores which may be used to 
do lots of calculations in parallel. 

### Usage

This is the help output:

```
hothothot [command] arg=value...

Commands:

    Info: displays GPU / CUDA info
    TimePoint: calculates the temperature for a given time and location
        Dimensions (DIM): 1, 2, or 3 (default 1)
        Location (L): location to measure (default 0)
        Device (DEV): set to CPU or GPU (default GPU)
        Time (T): time to measure (default 0)
        AmbientTemp (AMBIENT): ambient temperature (default 23)
        SourceTemp (SOURCE): temperature of heat source (default 100)
        Slices: the number of slices used (default 2500)
    Graph: displays GPU / CUDA info
    Help: displays this message

Example:

    hothothot TimePoint L=.7 T=10000000
```

### Dependencies

 - *vcpkg* for package management, see [docs](https://vcpkg.readthedocs.io/).
 - *gtest* for unit testing, installed with `vcpkg install gtest`
 - *CUDA* compiler, see below.
 
#### CUDA

You need the `nvcc` compiler which wraps GCC. The version available might 
not support the lastest GCC on your system, but you can install multiple 
versions and switch between them as needed. In my case, I'm on Ubuntu 20.04 
and will use the nvcc available in the repos as well as gcc 7.

```
sudo apt install nvidia-cuda-toolkit

sudo apt install build-essential
sudo apt -y install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
```

Using `sudo update-alternatives --config gcc` (and g++) you can set the 
versions used. Check them with `gcc --version`, `g++ --version`, and 
`nvcc --version`.
 
## Additional Reading

- [Cmake, googletest, and CLion setup using Ubuntu](https://raymii.org/s/tutorials/Cpp_project_setup_with_cmake_and_unit_tests.html)
