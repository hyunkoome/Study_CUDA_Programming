#include "./common.cpp"

// input parameters
unsigned num = 16 * 1024 * 1024; // num data
unsigned blocksize = 1024; // shared mem buf size

// CUDA kernel function
__global__ void kernelAdjDiff(float* b, const float* a, int num) {
	extern __shared__ float s_data[];
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
	// cuda availability check
	int deviceCount = 0;
	cudaGetDeviceCount( &deviceCount );
	CUDA_CHECK_ERROR();
	if (deviceCount == 0) {
		printf("%s: no available CUDA device found\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	cudaDeviceProp deviceProp;
	int driverVersion = 0;
	int runtimeVersion = 0;
	cudaGetDeviceProperties( &deviceProp, 0 ); // 0 is the default CUDA device
	cudaDriverGetVersion( &driverVersion );
	cudaRuntimeGetVersion( &runtimeVersion );
	printf("CUDA device \"%s\": driver ver %d.%d, runtime ver %d.%d, capability ver %d.%d\n",
	       deviceProp.name,
	       driverVersion / 1000, (driverVersion % 100) / 10,
	       runtimeVersion / 1000, (runtimeVersion % 100) / 10,
	       deviceProp.major, deviceProp.minor);
	printf("  max num of threads per block = %d\n", deviceProp.maxThreadsPerBlock);
	printf("  max dim size of a thread block (x,y,z) = (%d, %d, %d)\n",
	       deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
	       deviceProp.maxThreadsDim[2]);
	printf("  max dim size of a grid size (x,y,z) = (%d, %d, %d)\n",
	       deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
	       deviceProp.maxGridSize[2]);
	// my setting after device query
	blocksize = deviceProp.maxThreadsPerBlock;
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1 );
		break;
	case 3:
		num = procArg( argv[0], argv[1], 1 );
		blocksize = procArg( argv[0], argv[2], 32, deviceProp.maxThreadsPerBlock );
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
	ASSERT( dimBlock.x <= deviceProp.maxThreadsDim[0] );
	ASSERT( dimBlock.y <= deviceProp.maxThreadsDim[1] );
	ASSERT( dimBlock.z <= deviceProp.maxThreadsDim[2] );
	ASSERT( dimGrid.x <= deviceProp.maxGridSize[0] );
	ASSERT( dimGrid.y <= deviceProp.maxGridSize[1] );
	ASSERT( dimGrid.z <= deviceProp.maxGridSize[2] );
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
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */