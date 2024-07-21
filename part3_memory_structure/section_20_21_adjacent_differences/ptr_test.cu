#include "./common.cpp"

__device__ int my_g_var = 0; // CUDA device global variable
__constant__ int my_c_var = 1; // CUDA constant variable

// CUDA kernel function
__global__ void kernelTest(void) {
	__shared__ int my_s_var; // CUDA shared variable
	// initialize
	if (threadIdx.x == 0) {
		my_s_var = 2;
	}
	__syncthreads();
	// pointers
	int* ptr_to_global = &my_g_var;
	const int* ptr_to_constant = &my_c_var;
	int* ptr_to_shared = &my_s_var;
	// test
	*ptr_to_shared = *ptr_to_global;
	// test2
	int* ptr = nullptr;
	if (threadIdx.x % 2) {
		ptr = &my_g_var;
	} else {
		ptr = &my_s_var;
	}
	*ptr = *ptr_to_constant;
	__syncthreads();
	// print
	if (threadIdx.x == 0) {
		printf("global var = %d\n", my_g_var);
		printf("constant var = %d\n", my_c_var);
		printf("shared var = %d\n", my_s_var);
	}
}

// CUDA kernel function
__global__ void kernelTest2(void) {
	__shared__ int my_s_var; // CUDA shared variable
	// initialize
	if (threadIdx.x == 0) {
		my_s_var = 2;
	}
	__syncthreads();
	// pointers
	int* ptr_to_global = &my_g_var;
	const int* ptr_to_constant = &my_c_var;
	int* ptr_to_shared = &my_s_var;
	// test
	*ptr_to_shared = *ptr_to_global;
	// test2
	int* ptr = nullptr;
	ptr = &my_g_var;
	*ptr = *ptr_to_constant;
	ptr = &my_s_var;
	*ptr = *ptr_to_constant;
	__syncthreads();
	// print
	if (threadIdx.x == 0) {
		printf("global var = %d\n", my_g_var);
		printf("constant var = %d\n", my_c_var);
		printf("shared var = %d\n", my_s_var);
	}
}

int main(const int argc, const char* argv[]) {
	// CUDA kernel launch
	ELAPSED_TIME_BEGIN(0);
	kernelTest <<< 1, 32>>>( );
	cudaDeviceSynchronize();
	fflush(stdout);
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	ELAPSED_TIME_BEGIN(1);
	kernelTest2 <<< 1, 32>>>( );
	cudaDeviceSynchronize();
	fflush(stdout);
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
