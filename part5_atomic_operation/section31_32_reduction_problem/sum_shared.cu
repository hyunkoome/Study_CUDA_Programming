#include "./common.cpp"

// input parameters
const unsigned MAX_NUM = 16 * 1024 * 1024;
unsigned NUM = MAX_NUM; // num data
unsigned BLOCK_SIZE = 1024; // block size

__device__ float dev_sum = 0.0f;

// CUDA kernel function
__global__ void kernelSum( const float* pData, unsigned num ) {
	__shared__ float s_sum;
	unsigned gx = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (threadIdx.x == 0) {
		s_sum = 0.0f;
	}
	__syncthreads();
	if (gx < num) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 // CAUTION: atomicAdd_block(): CUDA arch 6.0 or later required.
#error CUDA arch 6.0 or later required. Use '-arch sm_60' or higher, to compile this program
#endif
		atomicAdd_block( &s_sum, pData[gx] );
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd( &dev_sum, s_sum );
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
	dim3 dimGrid(div_up(NUM, dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( NUM );
	ELAPSED_TIME_BEGIN(0);
	kernelSum <<< dimGrid, dimBlock>>>( dev_vecData, NUM );
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
