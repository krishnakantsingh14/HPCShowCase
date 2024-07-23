#include <iostream>
#include <cuda_runtime.h>

void printDeviceProperties(cudaDeviceProp &devProp) {
    std::cout << "Device Name: " << devProp.name << std::endl;
    std::cout << "Total Global Memory: " << devProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << devProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Registers per Block: " << devProp.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << devProp.warpSize << std::endl;
    std::cout << "Max Threads per Block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dim: ["
              << devProp.maxThreadsDim[0] << ", "
              << devProp.maxThreadsDim[1] << ", "
              << devProp.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "Max Grid Size: ["
              << devProp.maxGridSize[0] << ", "
              << devProp.maxGridSize[1] << ", "
              << devProp.maxGridSize[2] << "]" << std::endl;
    std::cout << "Clock Rate: " << devProp.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "Total Constant Memory: " << devProp.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "Multiprocessor Count: " << devProp.multiProcessorCount << std::endl;
    std::cout << "Compute Capability: " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "Memory Bus Width: " << devProp.memoryBusWidth << " bits" << std::endl;
    std::cout << "Memory Clock Rate: " << devProp.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "L2 Cache Size: " << devProp.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, dev);

        std::cout << "Device " << dev << " Properties:" << std::endl;
        printDeviceProperties(devProp);
        std::cout << std::endl;
    }

    return 0;
}
