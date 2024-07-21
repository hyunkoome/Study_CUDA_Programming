#include "./common.cpp"

__device__ int g_dim[5] = { 0, 1, 2, 3, 4 };
int mem_dim[5] = { 0, 1, 2, 3, 4 };

// CUDA kernel function
__global__ void kernelTest(int dim[]) {
	// print
	if (threadIdx.x == 0) {
		printf("dim = { %d, %d, %d, ... };\n", dim[0], dim[1], dim[2]);
	}
}

int main(const int argc, const char* argv[]) {
	// CUDA kernel launch
	ELAPSED_TIME_BEGIN(0);
	kernelTest <<< 1, 32>>>( g_dim );
	cudaDeviceSynchronize();
	fflush(stdout);
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	ELAPSED_TIME_BEGIN(1);
	kernelTest <<< 1, 32>>>( mem_dim );
	cudaDeviceSynchronize();
	fflush(stdout);
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
