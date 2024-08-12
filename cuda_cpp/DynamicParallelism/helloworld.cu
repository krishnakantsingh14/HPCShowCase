#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#include <cstdio>

__device__ int value{ -20 };

__global__ void fromGrandChild() {
    printf("Hello from Grand Child %d %d %d\n", value+threadIdx.x, threadIdx.x, blockIdx.x);

}

__device__ void fromChild() {
    printf("Hello from Child\n");
    value = 90;
    fromGrandChild<<<2,16>>>();  // Launch fromGrandChild kernel from the device
    //cudaDeviceSynchronize();  // Wait for the grandchild kernel to complete
}

__global__ void printfromcuda() {
    if (threadIdx.x == 0) {
        fromChild();  // Call the __device__ function that launches another kernel
    }
}

int main() {
    // Ensure your device supports dynamic parallelism before running this code
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.concurrentKernels) {
        std::cerr << "Device does not support concurrent kernel execution." << std::endl;
        return 1;
    }

    printfromcuda << <1, 32 >> > ();  // Launch the first kernel
    cudaDeviceSynchronize();  // Ensure the kernel completes before exiting

    return 0;
}
