#include <cuda_runtime.h>
#include <iostream>

// Kernel to square each element of the array
__global__ void squareKernel(float* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] *= d_array[idx];
    }
}

int main() {
    const int N = 1 << 20; // Total number of elements (1M elements)
    const int chunkSize = N / 4; // Process in chunks (4 chunks)
    const int bytes = N * sizeof(float);

    float* h_array; // Host array
    float* d_array; // Device array

    // Allocate pinned (page-locked) memory on the host
    cudaMallocHost((void**)&h_array, bytes);

    // Initialize the host array with random values
    for (int i = 0; i < N; ++i) {
        h_array[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_array, bytes);

    // Create CUDA streams
    cudaStream_t streams[4];
    for (int i = 0; i < 4; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch kernel and data transfers in a pipelined manner using pinned memory
    for (int i = 0; i < 4; ++i) {
        int offset = i * chunkSize;
        cudaMemcpyAsync(&d_array[offset], &h_array[offset], chunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        squareKernel << <chunkSize / 256, 256, 0, streams[i] >> > (&d_array[offset], chunkSize);
        cudaMemcpyAsync(&h_array[offset], &d_array[offset], chunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize streams
    for (int i = 0; i < 4; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Clean up
    cudaFree(d_array);
    cudaFreeHost(h_array); // Free pinned memory

    std::cout << "Array processing with pinned memory completed!" << std::endl;
    return 0;
}
