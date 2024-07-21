#include "./common.cpp"

// input parameters
unsigned num = 16 * 1024 * 1024; // num data
unsigned blocksize = 1024; // shared mem buf size

// CUDA kernel function
__global__ void kernelAdjDiff(float* b, const float* a, int num) {
	extern __shared__ float s_data[];  // area size not fixed at compile time
	register unsigned tx = threadIdx.x;
	register unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < num) {
		s_data[tx] = a[i];
		__syncthreads();
		if (tx > 0) {
			b[i] = s_data[tx] - s_data[tx - 1];
		} else if (i > 0) {
			b[i] = s_data[tx] - a[i - 1];
		} else { // i == 0
			b[i] = s_data[tx] - 0.0f;
		}
	}
}

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1 );
		break;
	case 3:
		num = procArg( argv[0], argv[1], 1 );
		blocksize = procArg( argv[0], argv[2], 32, 1024 );
		break;
	default:
		printf("usage: %s [num] [blocksize]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// host-side data
	float* vecA = nullptr;
	float* vecB = nullptr;
	try {
		vecA = new float[num];
		vecB = new float[num];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecA, num );
	// device-side data
	float* dev_vecA = nullptr;
	float* dev_vecB = nullptr;
	// allocate device memory
	ELAPSED_TIME_BEGIN(1);
	cudaMalloc( (void**)&dev_vecA, num * sizeof(float) );
	cudaMalloc( (void**)&dev_vecB, num * sizeof(float) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy( dev_vecA, vecA, num * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(blocksize, 1, 1);
	dim3 dimGrid(div_up(num, dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( num );
	ELAPSED_TIME_BEGIN(0);
	kernelAdjDiff <<< dimGrid, dimBlock, blocksize * sizeof(float)>>>( dev_vecB, dev_vecA, num );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( vecB, dev_vecB, num * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree( dev_vecA );
	cudaFree( dev_vecB );
	CUDA_CHECK_ERROR();
	// check the result
	float sumA = getSum( vecA, num );
	float sumB = getSum( vecB, num );
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printVec( "vecA", vecA, num );
	printVec( "vecB", vecB, num );
	// cleaning
	delete[] vecA;
	delete[] vecB;
	// done
	fflush(stdout);
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
