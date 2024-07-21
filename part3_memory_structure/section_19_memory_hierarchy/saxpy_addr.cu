#include "./common.cpp"

// fixed parameters
const unsigned vecSize = 64 * 1024 * 1024; // big-size elements

const float host_a[1] = { 1.234f };
float host_x[vecSize];
float host_y[vecSize];
float host_z[vecSize];

__constant__ float dev_a[1];
__device__ float dev_x[vecSize];
__device__ float dev_y[vecSize];
__device__ float dev_z[vecSize];


// CUDA kernel function
__global__ void kernelSAXPY( unsigned n ) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < n) {
		dev_z[i] = dev_a[0] * dev_x[i] + dev_y[i];
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
	void* ptr_a = nullptr;
	void* ptr_x = nullptr;
	void* ptr_y = nullptr;
	void* ptr_z = nullptr;

    // 기존의 cudaMemcpyToSymbol로 되어 있던 함수를 cudaGetSymbolAddress + cudaMemcpy 사용
	cudaGetSymbolAddress( &ptr_a, dev_a );
	cudaGetSymbolAddress( &ptr_x, dev_x );
	cudaGetSymbolAddress( &ptr_y, dev_y );
	cudaGetSymbolAddress( &ptr_z, dev_z );
	cudaMemcpy( ptr_a, host_a, 1 * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( ptr_x, host_x, vecSize * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( ptr_y, host_y, vecSize * sizeof(float), cudaMemcpyHostToDevice );
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
	cudaMemcpy( host_z, ptr_z, vecSize * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// check the result
	float sumX = getSum( host_x, vecSize );
	float sumY = getSum( host_y, vecSize );
	float sumZ = getSum( host_z, vecSize );
	float diff = fabsf( sumZ - (host_a[0] * sumX + sumY) );
	printf("SIZE = %d\n", vecSize);
	printf("a    = %f\n", host_a[0]);
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
