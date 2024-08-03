#include "./common.cpp"
#include <sm_60_atomic_functions.h>

// input parameters
const unsigned BLOCK_SIZE = 1024;
unsigned NUM = 64 * 1024 * 1024; // num of samplings

// CUDA device variables
__device__ unsigned long long dev_count = 0;
__device__ float dev_count_float = 0;

__device__ void myAtomicAdd(float* address, float value) {
	int oldval, newval, readback;
	oldval = __float_as_int(*address);
	newval = __float_as_int( __int_as_float(oldval) + value);
	while ((readback = atomicCAS( (int*)address, oldval, newval )) != oldval) {
		oldval = readback;
		newval = __float_as_int( __int_as_float(oldval) + value);
	}
}

// my atomicAdd implementation
template <typename TYPE>
__device__ void myAtomicAdd(TYPE* address, TYPE value) {
	TYPE oldval, newval, readback;
	oldval = *address;
	newval = oldval + value;
	while ((readback = atomicCAS( address, oldval, newval )) != oldval) {
		oldval = readback;
		newval = oldval + value;
	}
}

// CUDA kernel function
__global__ void kernelCount( unsigned num ) {
	__shared__ unsigned long long s_count;
	register unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadIdx.x == 0) {
		s_count = 0;
	}
	__syncthreads();
	if (i < num) {
		myAtomicAdd( &s_count, 1ULL );
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		myAtomicAdd( &dev_count, s_count );
	}
}

// CUDA kernel function
__global__ void kernelCountFloat( unsigned num ) {
	__shared__ float s_count;
	register unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadIdx.x == 0) {
		s_count = 0.0f;
	}
	__syncthreads();
	if (i < num) {
		myAtomicAdd( &s_count, 1.0f );
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		myAtomicAdd( &dev_count_float, s_count );
	}
}

int main( const int argc, const char* argv[] ) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		NUM = procArg( argv[0], argv[1], 1 );
		break;
	default:
		printf("usage: %s [num]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("NUM = %d\n", NUM);
	// CUDA kernel launch
	dim3 dimBlock( BLOCK_SIZE, 1, 1 );
	dim3 dimGrid( div_up(NUM, dimBlock.x), 1, 1 );
	ELAPSED_TIME_BEGIN(0);
	kernelCount <<< dimGrid, dimBlock>>>( NUM );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	ELAPSED_TIME_BEGIN(1);
	kernelCountFloat <<< dimGrid, dimBlock>>>( NUM );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(1);
	// copy to host from device
	unsigned long long count;
	cudaMemcpyFromSymbol( &count, dev_count, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost );
	float count_float;
	cudaMemcpyFromSymbol( &count_float, dev_count_float, sizeof(float), 0, cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	// check the result
	printf("num thread launched = %d\n", NUM);
	printf("count = %llu\n", count);
	printf("count float = %f\n", count_float);
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
