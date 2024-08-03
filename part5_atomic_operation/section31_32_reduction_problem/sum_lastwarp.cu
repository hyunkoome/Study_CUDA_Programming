#include "./common.cpp"

// input parameters
const unsigned MAX_NUM = 16 * 1024 * 1024;
unsigned NUM = MAX_NUM; // num data
unsigned BLOCK_SIZE = 1024; // block size

__device__ float dev_sum = 0.0f;

__device__ inline void warpSum(volatile float* s_data, unsigned tx) {
	s_data[tx] += s_data[tx + 32];
	s_data[tx] += s_data[tx + 16];
	s_data[tx] += s_data[tx + 8];
	s_data[tx] += s_data[tx + 4];
	s_data[tx] += s_data[tx + 2];
	s_data[tx] += s_data[tx + 1];
}

// CUDA kernel function
__global__ void kernelSum( const float* pData, unsigned strideElem ) {
	extern __shared__ float s_data[];
	register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
	register unsigned tx = threadIdx.x;
	// each thread loads one element from global to shared memory
	s_data[tx] = pData[gx] + pData[gx + strideElem];
	__syncthreads();
	// do reduction in the shared memory
	for (register unsigned stride = blockDim.x / 2; stride > 32; stride >>= 1) {
		if (tx < stride) {
			s_data[tx] += s_data[tx + stride];
		}
		__syncthreads();
	}
	// add the partial sum to the global answer
	if (tx < 32) {
		warpSum(s_data, tx);
		if (tx == 0) {
			atomicAdd( &dev_sum, s_data[0] );
		}
	}
}


int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		NUM = procArg( argv[0], argv[1], 1 );
		break;
	case 3:
		NUM = procArg( argv[0], argv[1], 1 );
		BLOCK_SIZE = procArg( argv[0], argv[2], 1, 1024 );
		break;
	default:
		printf("usage: %s [num] [blocksize]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("NUM = %d, BLOCK_SIZE = %d\n", NUM, BLOCK_SIZE);
	// host-side data
	float* vecData = nullptr;
	try {
		vecData = new float[NUM];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecData, NUM );
	// device-side data
	float* dev_vecData = nullptr;
	cudaMalloc( (void**)&dev_vecData, NUM * sizeof(float) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy( dev_vecData, vecData, NUM * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(div_up(NUM / 2, dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( NUM );
	unsigned strideElem = NUM / 2;
	ELAPSED_TIME_BEGIN(0);
	kernelSum <<< dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>( dev_vecData, strideElem );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	float sum;
	cudaMemcpyFromSymbol( &sum, dev_sum, sizeof(float), 0, cudaMemcpyDeviceToHost );
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// free device memory
	cudaFree( dev_vecData );
	CUDA_CHECK_ERROR();
	// check the result
	float sumData = getSum( vecData, NUM );
	printVec( "data", vecData, NUM );
	printf("sum(data) = %f from CPU processing\n", sumData);
	printf("sum =       %f from CUDA kernel\n", sum);
	printf("diff =      %f\n", fabsf(sumData - sum));
	printf("diff/num =  %f\n", fabsf(sumData - sum) / NUM);
	// cleaning
	delete[] vecData;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
