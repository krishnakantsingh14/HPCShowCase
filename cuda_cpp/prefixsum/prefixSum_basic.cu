#include<iostream>
#include<vector>
#include <cassert>
#include<cstdlib>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__host__ void prefixsum_cpu(int* input_array, int* output_array, int shape) {
    int acc = input_array[0];
    output_array[0] = acc;
    for (int i = 1; i < shape; ++i) {
        acc += input_array[i];
        output_array[i] = acc;
    }

}

__global__ void  kogge_stone_scan_kernel(int* X, int* Y, int *blockSum, int shape) {

    extern __shared__ int XY[]; //The shape should be equal to blockdim 
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < shape) {
        XY[threadIdx.x] = X[tid];
    }
    else {
        XY[threadIdx.x] = 0; // Handle cases where tid >= shape
    }
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;

        if (threadIdx.x >= stride && tid < shape)  {
            temp = XY[threadIdx.x - stride];
        }
        __syncthreads();
        XY[threadIdx.x] += temp;
        __syncthreads();

    }

    if (threadIdx.x == blockDim.x - 1) {
        blockSum[blockIdx.x] = XY[threadIdx.x];
    }

    if (tid < shape) {
        Y[tid] = XY[threadIdx.x];
    }
}

__global__ void reduced_blocks(int* Y, int *blockSum, int shape, int do_print =0) {
    //int B = blockIdx.x; 
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x>0 && tid < shape) {
        int sum = 0; 
        for (int i = 0; i < blockIdx.x; ++i) {
            sum += blockSum[i];
        }
        Y[tid] += sum;
        if (do_print ==1) {
            printf("%d, %d %d \n", sum, blockIdx.x, blockSum[blockIdx.x-1]);
        }
    }

}


__host__ void check_array(int* first_array, int* second_array, int size) {

    for (int i = 0; i < size; ++i) {
        if (first_array[i] != second_array[i]) {
            std::cerr << "Arrays differ at index " << i << ": "
                << "arr1[" << i << "] = " << first_array[i] << ", "
                << "arr2[" << i << "] = " << second_array[i] << std::endl;
        }
    }
    std::cout << "SUCCESS" << std::endl;
}



__host__ void initialize_array(int* inputarray, int size) {

    for (int i = 0; i < size; ++i) {
        inputarray[i] = 1;// rand() % 100; //random numbers from 0 to 100;
    }
}


int main() {
    const int SHAPE = 1 << 8;  // 256
    const size_t bytesOfArray = SHAPE * sizeof(int);

    // Allocate and initialize arrays
    int* inp_array = nullptr;
    int* out_cpu_array = new int[SHAPE];
    int* out_gpu_array = nullptr;

    cudaMallocManaged(&inp_array, bytesOfArray);
    cudaMallocManaged(&out_gpu_array, bytesOfArray);

    initialize_array(inp_array, SHAPE);

    // Initialize output arrays
    std::fill(out_cpu_array, out_cpu_array + SHAPE, 0);
    std::fill(out_gpu_array, out_gpu_array + SHAPE, 0);

    // CPU Prefix Sum
    prefixsum_cpu(inp_array, out_cpu_array, SHAPE);
    std::cout << "CPU computation done." << std::endl;

    // GPU Prefix Sum using Kogge-Stone scan
    const int blockSize = 128;
    const int numBlocks = (SHAPE + blockSize - 1) / blockSize;
    const size_t sharedMemSize = blockSize * sizeof(int);

    int* block_sums = nullptr;
    cudaMalloc(&block_sums, numBlocks * sizeof(int));

    // Launching the kernels
    kogge_stone_scan_kernel<<<numBlocks, blockSize, sharedMemSize>>>(inp_array, out_gpu_array, block_sums, SHAPE);
    reduced_blocks<<<numBlocks, blockSize>>>(out_gpu_array, block_sums, SHAPE);

    // Synchronize to wait for the GPU to finish
    cudaDeviceSynchronize();
    std::cout << "GPU computation done." << std::endl;

    // Check the results
    check_array(out_cpu_array, out_gpu_array, SHAPE);

    // Clean up resources
    cudaFree(inp_array);
    cudaFree(out_gpu_array);
    cudaFree(block_sums);
    delete[] out_cpu_array;

    return 0;
}