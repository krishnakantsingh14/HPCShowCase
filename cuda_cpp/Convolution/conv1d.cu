#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<iostream>

#define FILTER_SIZE 5
#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM 28 //we assume blockDim is 32 and radius is 2
__constant__ int sFilter[FILTER_SIZE] ;

void errorCheck(int *firstArray, int *secondArray, int *filter, int r, int width) {
	int temp_chk{ 0 };
	int SUCCESS_count = 0;
	for (int i = 0 + r; i < width - r; ++i) {
		temp_chk = 0;
		for (int j = 0; j < 5; ++j) {

			temp_chk += firstArray[i + j - r] * filter[j];
		}
		if (secondArray[i] != temp_chk) {
			std::cout << "ERROR" << std::endl;
		}
		else {
			SUCCESS_count++;
		}


	}
	std::cout << SUCCESS_count << " out of " << width - 4 << std::endl;

}



__global__ void convolution_1d_shared_memory_kernel(int* N, int* P, int r, int width, int height) {
	//height will be ignored, height parameter is only for consistency purpose (with2d kernel).
	// Note, we are not passing filter as input, we will use Filter using __shared__ c
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int temp_p = 0;
	if ((tid - r) >= 0 && (tid + r) < width) {
		for (int i = 0; i < 2 * r + 1; ++i) {
			temp_p += N[tid - r + i] * sFilter[i];
		}
	}
	if (tid < width) {
		P[tid] = temp_p;
	}
}


__global__ void convolution_1d_basic_kernel(int* N, int* F, int* P, int r, int width, int height) {
	//height will be ignored, height parameter is only for consistency purpose (with2d kernel).
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int temp_p = 0;
	if ((tid - r) >= 0 && (tid + r) < width) {
		for (int i = 0; i < 2 * r + 1; ++i) {
			temp_p += N[tid - r + i] * F[i];
		}
	}
	if (tid < width) {
		P[tid] = temp_p;
	}
}

__global__ void convolution_1d_tiled(int *N, int *P,int r, int width) {
	
	int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x -  FILTER_RADIUS;
	extern __shared__ int s_array[IN_TILE_DIM];
    
	if (col >=0 && col < width) {
		s_array[col] = N[col];
	}
	else {
		s_array[col] =0;
	}
	__syncthreads();

	int tileCol = threadIdx.x - FILTER_RADIUS;
	if (col >=0 && col < width) {
		if (tileCol >=0 && tileCol < width) {
			int tmp_sum = 0;
			
			for (int i=0; i<FILTER_SIZE; ++i) {
				tmp_sum += s_array[tileCol+i]*sFilter[i];
			}
			P[col] = tmp_sum;
		}
	}
}

int main() {
	int width = 1024 * 10;
	int r = 2;
	int * input1DArray = new int[width];
	int filter[5]{ 1,2,0,2,1 };
	int* result = new int[width];
	int* d_vec, int* p, *F;
	int temp_chk = 0;
	cudaMalloc(&d_vec, width * sizeof(int));
	cudaMalloc(&p, width * sizeof(int));
	cudaMalloc(&F, 5 * sizeof(int));


	for (int i = 0; i < width; ++i) {
		input1DArray[i] = i;
	}
	cudaMemcpy(d_vec, input1DArray, width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(F, filter, 5 * sizeof(int), cudaMemcpyHostToDevice) ;

	convolution_1d_basic_kernel << < 10, 1024 >> > (d_vec, F, p, 2, width, 0);

	cudaMemcpy(result, p, width * sizeof(int), cudaMemcpyDeviceToHost);

	errorCheck(input1DArray, result, filter, 2, 10 * 1024);
	for (int i = 0; i < 5; ++i) {
		filter[i] = -2 + i;
	}
	//launch1DconvolutionSharedMemory(filter);
	
	cudaMemcpyToSymbol(sFilter, filter, 5 * sizeof(int));
	convolution_1d_shared_memory_kernel<<< 10, 1024 >> > (d_vec, p, 2, width, 0);


	cudaMemcpy(result, p, width * sizeof(int), cudaMemcpyDeviceToHost);
	errorCheck(input1DArray, result, filter, 2, 10 * 1024);

	convolution_1d_tiled<<<10, 1024 >>> (d_vec, p, 2, width);
	cudaMemcpy(result, p, width * sizeof(int), cudaMemcpyDeviceToHost);
	errorCheck(input1DArray, result, filter, 2, 10 * 1024);

	for (int i = 0; i < 5; ++i) {
		std::cout << filter[i] << std::endl;;
	}
}