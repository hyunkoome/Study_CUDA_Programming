#include "./common.cpp"

// fixed parameters
const unsigned vecSize = 64 * 1024 * 1024; // big-size elements
const float host_a = 1.234f;

float host_x[vecSize];
float host_y[vecSize];
float host_z[vecSize];

// 기존 코드 처럼 cudaMalloc 로 변수 잡지 않고, 변수로 잡음
__constant__ float dev_a = 1.234f;
__device__ float dev_x[vecSize];
__device__ float dev_y[vecSize];
__device__ float dev_z[vecSize];


// CUDA kernel function
__global__ void kernelSAXPY( unsigned n ) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < n) {
		dev_z[i] = dev_a * dev_x[i] + dev_y[i];
	}
}

int main( const int argc, const char* argv[] ) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	default:
		printf("usage: %s\n", argv[0]); // everything fixed !
		exit( EXIT_FAILURE );
		break;
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( host_x, vecSize );
	setNormalizedRandomData( host_y, vecSize );

	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpyToSymbol( dev_x, host_x, sizeof(host_x) );
	cudaMemcpyToSymbol( dev_y, host_y, sizeof(host_y) );
	CUDA_CHECK_ERROR();

	// CUDA kernel launch
	dim3 dimBlock( 1024, 1, 1 );
	dim3 dimGrid( (vecSize + dimBlock.x - 1) / dimBlock.x, 1, 1 );
	CUDA_PRINT_CONFIG( vecSize );
	ELAPSED_TIME_BEGIN(0);
	kernelSAXPY <<< dimGrid, dimBlock>>>( vecSize );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();

	// copy to host from device
	cudaMemcpyFromSymbol( host_z, dev_z, sizeof(host_z) );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);

	// check the result
	float sumX = getSum( host_x, vecSize );
	float sumY = getSum( host_y, vecSize );
	float sumZ = getSum( host_z, vecSize );
	float diff = fabsf( sumZ - (host_a * sumX + sumY) );
	printf("SIZE = %d\n", vecSize);
	printf("a    = %f\n", host_a);
	printf("sumX = %f\n", sumX);
	printf("sumY = %f\n", sumY);
	printf("sumZ = %f\n", sumZ);
	printf("diff(sumZ, a*sumX+sumY) =  %f\n", diff);
	printf("diff(sumZ, a*sumX+sumY)/SIZE =  %f\n", diff / vecSize);
	printVec( "vecX", host_x, vecSize );
	printVec( "vecY", host_y, vecSize );
	printVec( "vecZ", host_z, vecSize );
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
