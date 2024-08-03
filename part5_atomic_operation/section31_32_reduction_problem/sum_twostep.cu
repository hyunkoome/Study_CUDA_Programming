#include "./common.cpp"

// input parameters
const unsigned MAX_NUM = 16 * 1024 * 1024;
unsigned NUM = MAX_NUM; // num data
unsigned BLOCK_SIZE = 1024; // block size

__device__ inline void warpSum(volatile float* s_data, unsigned tx) {
	// initialize the data for every lane
	float value = s_data[tx] + s_data[tx + 32];
	// shuffle the data
	for (unsigned i = 16; i >= 1; i >>= 1) {
		value += __shfl_xor_sync( 0xFFFFFFFF, value, i, 32 );
	}
	s_data[tx] = value;
}

// CUDA kernel function
__global__ void kernelSum( float* pSum, const float* pData, unsigned strideElem ) {
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
			pSum[blockIdx.x] = s_data[0];
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
	float* vecSum = nullptr;
	try {
		vecData = new float[NUM];
		vecSum = new float[1];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecData, NUM );
	// device-side data
	float* dev_vecData = nullptr;
	float* dev_vecSum1 = nullptr;
	float* dev_vecSum2 = nullptr;
	cudaMalloc( (void**)&dev_vecData, NUM * sizeof(float) );
	cudaMalloc( (void**)&dev_vecSum1, (NUM / 2 / BLOCK_SIZE) * sizeof(float) );
	cudaMalloc( (void**)&dev_vecSum2, (NUM / 2 / BLOCK_SIZE) * sizeof(float) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy( dev_vecData, vecData, NUM * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemset( dev_vecSum1, 0, (NUM / 2 / BLOCK_SIZE) * sizeof(float) );
	cudaMemset( dev_vecSum2, 0, (NUM / 2 / BLOCK_SIZE) * sizeof(float) );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(div_up(NUM / 2, dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( NUM );
	unsigned strideElem = NUM / 2;
	ELAPSED_TIME_BEGIN(0);
	kernelSum <<< dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>( dev_vecSum1, dev_vecData, strideElem );
	dim3 dimGrid2(div_up(dimGrid.x / 2, dimBlock.x), 1, 1);
	printf("dimGrid2  = %d * %d * %d\n", dimGrid2.x, dimGrid2.y, dimGrid2.z);
	strideElem = dimGrid.x / 2;
	kernelSum <<< dimGrid2, dimBlock, dimBlock.x * sizeof(float)>>>( dev_vecSum2, dev_vecSum1, strideElem );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( vecSum, dev_vecSum2, dimGrid2.x * sizeof(float), cudaMemcpyDeviceToHost );
	float sum = getSum( vecSum, dimGrid2.x );
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// free device memory
	cudaFree( dev_vecData );
	cudaFree( dev_vecSum1 );
	cudaFree( dev_vecSum2 );
	CUDA_CHECK_ERROR();
	// check the result
	float sumData = getSum( vecData, NUM );
	printVec( "data", vecData, NUM );
	printf("sum(data) = %f from CPU processing\n", sumData);
	printf("sum =       %f from CUDA kernel\n", sum);
	printf("diff     =  %f\n", fabsf(sumData - sum));
	printf("diff/num =  %f\n", fabsf(sumData - sum) / NUM);
	// cleaning
	delete[] vecData;
	delete[] vecSum;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
