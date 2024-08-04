#include "./common.cpp"
#include <limits.h>

// input parameters
unsigned num = 16 * 1024 * 1024; // maximum num of inputs
unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)
unsigned BLOCK_SIZE = 1024; // block size

__device__ unsigned dev_index = UINT_MAX; // initialized to MAX

// CUDA kernel function
__global__ void kernelSearch(const unsigned* pData, unsigned num, unsigned target) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < num) {
		if (pData[i] == target) {
			atomicMin( &dev_index, i );
		}
	}
}

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1024 );
		break;
	case 3:
		num = procArg( argv[0], argv[1], 1024 );
		BLOCK_SIZE = procArg( argv[0], argv[2], 32, 1024 );
		break;
	default:
		printf("usage: %s [num] [blocksize]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("num = %u, BLOCK_SIZE = %u\n", num, BLOCK_SIZE);
	printf("bound = %u\n", bound);
	// host-side data
	unsigned* vecData = nullptr;
	try {
		vecData = new unsigned[num];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data to A and B
	srand( 0 );
	setRandomData<unsigned>( vecData, num, bound );
	unsigned targetValue = vecData[num - 1];
	// device-side data
	unsigned* dev_vecData = nullptr;
	cudaMalloc( (void**)&dev_vecData, num * sizeof(unsigned) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy( dev_vecData, vecData, num * sizeof(unsigned), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel call
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(div_up(num, dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( num );
	ELAPSED_TIME_BEGIN(0);
	kernelSearch <<< dimGrid, dimBlock>>>( dev_vecData, num, targetValue );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	unsigned index = UINT_MAX;
	cudaMemcpyFromSymbol( &index, dev_index, sizeof(unsigned), 0, cudaMemcpyDeviceToHost );
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// free device memory
	cudaFree( dev_vecData );
	CUDA_CHECK_ERROR();
	// check the result
	if (index >= num) {
		printf("NOT FOUND: target value '%u' not found\n", targetValue);
	} else {
		printf("FOUND: vecData[%d] = %d\n", index, vecData[index]);
	}
	printVec( "vecData", vecData, num );
	// cleaning
	delete[] vecData;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
