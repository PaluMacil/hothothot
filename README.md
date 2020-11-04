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
    Graph: outputs data to csv for the given point (same parameters as TimePoint)
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

### Analysis

When performing benchmarking with CUDA, I experienced no speedup and even 
a slowdown before attempting omptimizations.

#### Checking Correctness

I used unit tests by way of Google's gtest library. This is a leading test 
framework along with Boost's test framework. While I also tested the configuration
option parsing, the type tests below are what let me know both CPU and GPU runs of
the application matched for a known problem. The math below is similar to a known 
solution except that the solution I referenced treated the first position in the 1D
slice as the source temperature position whereas I used position -1 to reference 
that value. For the unit tests to pass, both CPU and GPU calculations had to return 
the known answer.

```
TEST(Calculator, ExecCPU) {
    std::vector<std::string> args;
    args.emplace_back("TIME=10000000");
    args.emplace_back("LOCATION=.7");
    args.emplace_back("TIMEPOINT");
    config::Configuration conf(args);
    Calculator calc(conf);
    auto answer = calc.exec();

    EXPECT_FLOAT_EQ(answer, 85.5276794);
}

TEST(Calculator, ExecGPU) {
    std::vector<std::string> args;
    args.emplace_back("TIME=10000000");
    args.emplace_back("LOCATION=.7");
    args.emplace_back("TIMEPOINT");
    args.emplace_back("DEVICE=GPU");
    config::Configuration conf(args);
    Calculator calc(conf);
    auto answer = calc.exec();

    EXPECT_FLOAT_EQ(answer, 85.5276794);
}
```

#### Performance Evaluation

Before optimizing, my initial provably correct run on the GPU was slower than the CPU,
taking 29.5% more time on average than the CPU. This was using only global memory 
with no optimizations. Attempts to improve this time are discussed in the next section.

| Run #         | CPU      | GPU      |
|---------------|----------|----------|
| 1             | 19.8001  | 28.1022  |
| 2             | 19.5416  | 27.7406  |
| 3             | 19.5416  | 27.7606  |
| 4             | 19.8165  | 28.1595  |
| 5             | 19.4779  | 28.0528  |
| 6             | 19.8165  | 28.4575  |
| 7             | 19.7752  | 28.0171  |
| 8             | 19.5277  | 27.5968  |
| 9             | 19.5061  | 27.5574  |
| 10            | 19.6681  | 27.4034  |
| Average (sec) | 19.64713 | 27.88479 |

#### Optimizing

Initially, I tried using shared memory. Shared memory is one to two 
orders of magnitude faster than global memory. The following attempt 
failed shown below because each block will have a different set of shared memory,
so the neighbor values on each side will be inaccurate. I could have 
synchronized the entire device instead of the block, experimented 
with getting block neighboring values from the global memory, etc, but since 
my use of shared memory showed no runtime speed improvement, I knew additional 
synchronization points would only make it worse.

```
__global__ void execGPU_d(int n, float sourceTemp, const float *currentArray, float *nextArray) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        extern __shared__ float currentBlock[];
        currentBlock[i] = currentArray[i];
        __syncthreads();

        auto idxBefore = i - 1;
        auto valBefore =
                idxBefore == -1 ?
                // if before the start of the array, use temperature of the heat source
                sourceTemp :
                currentBlock[idxBefore];
        auto idxAfter = i + 1;
        auto valAfter =
                idxAfter == n ?
                // if the last element, use own temperature
                currentBlock[i] :
                currentBlock[idxAfter];
        nextArray[i] = (valBefore + valAfter) / 2;
    }
}
```
 
## Additional Reading

- [Cmake, googletest, and CLion setup using Ubuntu](https://raymii.org/s/tutorials/Cpp_project_setup_with_cmake_and_unit_tests.html)
