#include "./common.cpp"

// input parameters
unsigned num = 32 * 1024 * 1024; // num of data

// CUDA kernel function
__global__ void kernelClear( float* dst, unsigned num ) {
	int gx = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	dst[gx] = 0.0f;
}

int main( const int argc, const char* argv[] ) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1024 );
		break;
	default:
		printf("usage: %s [num]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("num=%u\n", num);
	// host-side data
	float* alpha = new float[num];
	float* bravo = new float[num];
	setNormalizedRandomData( alpha, num );
	setNormalizedRandomData( bravo, num );
	// device-side data
	float* dev_alpha = nullptr;
	float* dev_bravo = nullptr;
	// allocate device memory
	cudaMalloc( (float**)&dev_alpha, num * sizeof(float) );
	cudaMalloc( (float**)&dev_bravo, num * sizeof(float) );
	cudaMemcpy( dev_alpha, alpha, num * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_bravo, bravo, num * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// show the original contents
	cudaMemcpy( alpha, dev_alpha, num * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( bravo, dev_bravo, num * sizeof(float), cudaMemcpyDeviceToHost );
	printVec( "alpha = ", alpha, num );
	printVec( "bravo = ", bravo, num );
	// CUDA kernel launch
	dim3 dimBlock( 1024, 1, 1 );
	dim3 dimGrid( div_up(num, dimBlock.x), 1, 1 );
	ELAPSED_TIME_BEGIN(0);
	kernelClear <<< dimGrid, dimBlock>>>( dev_alpha, num );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// CUDA memset
	ELAPSED_TIME_BEGIN(1);
	cudaMemset( dev_bravo, 0, num * sizeof(float) );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( alpha, dev_alpha, num * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( bravo, dev_bravo, num * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	// check the result
	printVec( "alpha = ", alpha, num );
	printVec( "bravo = ", bravo, num );
	// free device memory
	cudaFree( dev_alpha );
	cudaFree( dev_bravo );
	CUDA_CHECK_ERROR();
	// cleaning
	delete[] alpha;
	delete[] bravo;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
