#include "./common.cpp"
#include <limits.h>

// input parameters
unsigned num = 16 * 1024 * 1024; // maximum num of inputs
unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)
unsigned BLOCK_SIZE = 1024; // block size
unsigned stride = 512 * 1024; // stride between threads

// CUDA kernel function
__global__ void kernelSearch(const unsigned* pData, unsigned num, unsigned target, unsigned* pIndex, unsigned stride) {
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	for (unsigned j = index; j < num; j += stride) {
		if (pData[j] == target) {
			pIndex[index] = j;
			index++;
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
	case 4:
		num = procArg( argv[0], argv[1], 1024 );
		BLOCK_SIZE = procArg( argv[0], argv[2], 32, 1024 );
		stride = procArg( argv[0], argv[3], 32 );
		break;
	default:
		printf("usage: %s [num] [blocksize] [stride]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("num = %u, BLOCK_SIZE = %u, stride = %u\n", num, BLOCK_SIZE, stride);
	printf("bound = %u\n", bound);
	// host-side data
	unsigned* vecData = nullptr;
	unsigned* vecIndex = nullptr;
	try {
		vecData = new unsigned[num];
		vecIndex = new unsigned[stride];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data to A and B
	srand( 0 );
	setRandomData<unsigned>( vecData, num, bound );
	unsigned targetValue = vecData[num - 1];
	printf("targetValue = %u\n", targetValue);
	memset( vecIndex, 0xFF, stride * sizeof(unsigned) );
	// device-side data
	unsigned* dev_vecData = nullptr;
	unsigned* dev_vecIndex = nullptr;
	cudaMalloc( (void**)&dev_vecData, num * sizeof(unsigned) );
	cudaMalloc( (void**)&dev_vecIndex, stride * sizeof(unsigned) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy( dev_vecData, vecData, num * sizeof(unsigned), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_vecIndex, vecIndex, stride * sizeof(unsigned), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel call
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(div_up(stride, dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( num );
	ELAPSED_TIME_BEGIN(0);
	kernelSearch <<< dimGrid, dimBlock>>>( dev_vecData, num, targetValue, dev_vecIndex, stride );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( vecIndex, dev_vecIndex, stride * sizeof(unsigned), cudaMemcpyDeviceToHost );
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// free device memory
	cudaFree( dev_vecData );
	cudaFree( dev_vecIndex );
	CUDA_CHECK_ERROR();
	// check the result
	unsigned found = 0;
	for (unsigned i = 0; i < stride; ++i) {
		if (vecIndex[i] < num) {
			unsigned index = vecIndex[i];
			printf("vecData[%d]= %d\n", index, vecData[index]);
			found++;
		}
	}
	printf("%d locations are found\n", found);
	printVec( "vecIndex", vecIndex, found );
	printVec( "vecData", vecData, num );
	// cleaning
	delete[] vecData;
	delete[] vecIndex;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
