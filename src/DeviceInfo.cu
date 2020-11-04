// DeciveInfo.cu
// Dan Wolf

#include "DeviceInfo.cuh"
#include <iostream>

void DeviceInfo::print() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        std::printf("Device Number: %d\n", i);
        std::printf("  Device name: %s\n", prop.name);
        std::printf("  Memory Clock Rate (KHz): %d\n",
                    prop.memoryClockRate);
        std::printf("  Memory Bus Width (bits): %d\n",
                    prop.memoryBusWidth);
        std::printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                    2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}