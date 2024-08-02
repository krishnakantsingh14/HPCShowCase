#include<cstdlib>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>
#include <cassert>

__global__ void histogram(int* d_inputarray, int* d_bins, int n_elements, int n_bins, int div) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_elements) {
        int bin = d_inputarray[tid] / div;
        atomicAdd(&d_bins[bin], 1);
    }
}


__global__ void histogramSharedMemory(int* d_inputarray, int* d_bins, int n_elements, int n_bins, int div) {

    extern __shared__ int binBlocks[]; //bins in each block; 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndex = threadIdx.x;
    if (threadIndex < n_bins ) {
        binBlocks[threadIndex] = 0;
    }

    __syncthreads();

    if (tid < n_elements) {
        int bin = d_inputarray[tid] / div;
        atomicAdd(&binBlocks[bin], 1);
    }

    __syncthreads();
    if (threadIndex < n_bins) {

        atomicAdd(&d_bins[threadIndex], binBlocks[threadIndex]);
    }
}

void initialize_array(int* inputarray, int size) {

    for (int i = 0; i < size; ++i) {
        inputarray[i] = rand() % 100; //random numbers from 0 to 100;
    }
}


void check_array(int* input_array, int size, int* output_bin, int n_bins, int div) {
    int* tmp_bins = new int[n_bins];
    for (int i = 0; i < n_bins; ++i) {
        tmp_bins[i] = 0;
    }

    for (int i = 0; i < size; ++i) {
        int tmp_index = input_array[i] / div;
        tmp_bins[tmp_index] += 1;
    }

    for (int i = 0; i < n_bins; ++i) {
        assert(tmp_bins[i] ==  output_bin[i]) ;
    }

    for (int i = 0; i < n_bins; ++i) {
        std::cout <<" CPU:  "  << tmp_bins[i]  << " GPU: " << output_bin[i] << std::endl;
    }
    std::cout << "SUCCESS" << std::endl;

    delete tmp_bins;
}


int main() {
    int n_elements{ 1 << 20 }; //2^20 elemetes
    size_t nbytes{ n_elements * sizeof(int) };

    int n_bins = 10;
    int div = (100 + n_bins - 1) / n_bins;
    size_t nbinsbytes{ n_bins * sizeof(int) };

    int* elements, * bins;
    cudaMallocManaged(&elements, nbytes);
    cudaMallocManaged(&bins, nbinsbytes);

    initialize_array(elements, n_elements);
    initialize_array(bins, n_bins);
    for (int i = 0; i < n_bins; ++i) {
        bins[i] = 0;
    }

    int NTHREADS{ 256 };
    int BLOCKS{ (n_elements + NTHREADS - 1) / NTHREADS };
    std::cout << "Before kernel Launch" << std::endl;

    //histogram << <BLOCKS, NTHREADS >> > (elements, bins, n_elements, n_bins, div);
    histogramSharedMemory << <BLOCKS, NTHREADS, nbinsbytes >> > (elements, bins, n_elements, n_bins, div);
    cudaDeviceSynchronize();

    check_array(elements, n_elements, bins, n_bins, div);
}