#include<iostream>
#include<vector>

__host__ void prefixsum_cpu(int *input_array, int *output_array, int shape) {
    int acc = input_array[0];
    output_array[0] = acc;
    for (int i=1; i<shape; ++i) {
        acc += input_array[i];
        output_array[i] = acc ;
    }

}

__global__ void  kogge_stone_scan_kernel(int *X, int *Y, int shape) {

    __shared__ int XY[blockDim.x];
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x; 
    if (tid<shape) {
        XY[threadIdx.x] = X[tid];
    }

    for (unsigned int stride =1; stride < blockDim.x ; stride *= 2 ) {
        if (threadIdx.x >= stride) {
            XY[threadIdx.x] += XY[threadIdx.x-stride]; 
        }  
        
    }
    Y[tid] = XY[threadIdx.x];

}

int main() {
    int * inp_array = new int [5]{1,10,20,3,4};
    int * out_array = new int [5]{0,0,0,0,0};

    prefixsum_cpu(inp_array, out_array, 5);
    for (int i=0; i<5; ++i) {
        std::cout << out_array[i] << std::endl;
    }
}