#include<cstdlib>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>
#include <cassert>

__global__ void histogram(int* d_inputarray, int* d_bins, int n_elements, int n_bins, int div) {
    // Calculate the global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Ensure the thread is within bounds of the input array
    if (tid < n_elements) {
        // Calculate the bin index for the current element
        int bin = d_inputarray[tid] / div;
        
        // Atomically increment the appropriate bin in the global histogram
        atomicAdd(&d_bins[bin], 1);
    }
}

__global__ void histogramSharedMemory(int* d_inputarray, int* d_bins, int n_elements, int n_bins, int div) {
    // Declare shared memory array to hold bins for each block
    extern __shared__ int binBlocks[];
    
    // Calculate the global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Get the thread index within the block
    int threadIndex = threadIdx.x;
    
    // Initialize the shared memory bins to zero
    if (threadIndex < n_bins) {
        binBlocks[threadIndex] = 0;
    }

    // Synchronize threads to ensure all bins are initialized
    __syncthreads();

    // Check if the thread is within bounds of the input array
    if (tid < n_elements) {
        // Calculate the bin index for the current element
        int bin = d_inputarray[tid] / div;
        
        // Atomically increment the appropriate bin in shared memory
        atomicAdd(&binBlocks[bin], 1);
    }

    // Synchronize threads to ensure all updates to shared memory are done
    __syncthreads();
    
    // Aggregate the shared memory bins into the global bins
    if (threadIndex < n_bins) {
        atomicAdd(&d_bins[threadIndex], binBlocks[threadIndex]);
    }
}




__host__ void initialize_array(int* inputarray, int size) {

    for (int i = 0; i < size; ++i) {
        inputarray[i] = rand() % 100; //random numbers from 0 to 100;
    }
}


__host__ void check_array(int* input_array, int size, int* output_bin, int n_bins, int div) {
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
    const int n_elements{ 1 << 20 }; //2^20 elemetes
    const int n_bins = 10; // Number of bins
    const int div = (100 + n_bins - 1) / n_bins;
    const size_t nbytes{ n_elements * sizeof(int) };
    const size_t nbinsbytes{ n_bins * sizeof(int) };

    int* elements;
    int * bins;
    cudaMallocManaged(&elements, nbytes); 
    cudaMallocManaged(&bins, nbinsbytes);

    initialize_array(elements, n_elements);
    initialize_array(bins, n_bins);

    for (int i = 0; i < n_bins; ++i) {
        bins[i] = 0;
    }

    const int NTHREADS{ 256 };
    const int BLOCKS{ (n_elements + NTHREADS - 1) / NTHREADS };


    //histogram << <BLOCKS, NTHREADS >> > (elements, bins, n_elements, n_bins, div);
    histogramSharedMemory << <BLOCKS, NTHREADS, nbinsbytes >> > (elements, bins, n_elements, n_bins, div);
    cudaDeviceSynchronize();

    check_array(elements, n_elements, bins, n_bins, div);
    cudaFree(elements);
    cudaFree(bins);
}