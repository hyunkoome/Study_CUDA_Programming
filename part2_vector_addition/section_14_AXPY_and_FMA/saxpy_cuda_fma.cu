#include "./common.cpp"

// input parameters
unsigned vecSize = 256 * 1024 * 1024; // big-size elements
float saxpy_a = 1.234f;

// CUDA kernel function
__global__ void kernelSAXPY( float* z, const float a, const float* x, const float* y, unsigned n ) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < n) {
		z[i] = fmaf( a, x[i], y[i] );
	}
}

int main( const int argc, const char* argv[] ) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		vecSize = procArg( argv[0], argv[1], 1 );
		break;
	case 3:
		vecSize = procArg( argv[0], argv[1], 1 );
		saxpy_a = procArg<float>( argv[0], argv[2] );
		break;
	default:
		printf("usage: %s [num] [a]\n", argv[0]);
		exit( EXIT_FAILURE );
		break;
	}
	// host-side data
	float* vecX = nullptr;
	float* vecY = nullptr;
	float* vecZ = nullptr;
	try {
		vecX = new float[vecSize];
		vecY = new float[vecSize];
		vecZ = new float[vecSize];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(1);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecX, vecSize );
	setNormalizedRandomData( vecY, vecSize );
	// device-side data
	float* dev_vecX = nullptr;
	float* dev_vecY = nullptr;
	float* dev_vecZ = nullptr;
	// allocate device memory
	cudaMalloc( (void**)&dev_vecX, vecSize * sizeof(float) );
	cudaMalloc( (void**)&dev_vecY, vecSize * sizeof(float) );
	cudaMalloc( (void**)&dev_vecZ, vecSize * sizeof(float) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy( dev_vecX, vecX, vecSize * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_vecY, vecY, vecSize * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock( 1024, 1, 1 );
	dim3 dimGrid( (vecSize + dimBlock.x - 1) / dimBlock.x, 1, 1 );
	CUDA_PRINT_CONFIG( vecSize );
	ELAPSED_TIME_BEGIN(0);
	kernelSAXPY <<< dimGrid, dimBlock>>>( dev_vecZ, saxpy_a, dev_vecX, dev_vecY, vecSize );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( vecZ, dev_vecZ, vecSize * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree( dev_vecX );
	cudaFree( dev_vecY );
	cudaFree( dev_vecZ );
	CUDA_CHECK_ERROR();
	// check the result
	float sumX = getSum( vecX, vecSize );
	float sumY = getSum( vecY, vecSize );
	float sumZ = getSum( vecZ, vecSize );
	float diff = fabsf( sumZ - (saxpy_a * sumX + sumY) );
	printf("SIZE = %d\n", vecSize);
	printf("a    = %f\n", saxpy_a);
	printf("sumX = %f\n", sumX);
	printf("sumY = %f\n", sumY);
	printf("sumZ = %f\n", sumZ);
	printf("diff(sumZ, a*sumX+sumY) =  %f\n", diff);
	printf("diff(sumZ, a*sumX+sumY)/SIZE =  %f\n", diff / vecSize);
	printVec( "vecX", vecX, vecSize );
	printVec( "vecY", vecY, vecSize );
	printVec( "vecZ", vecZ, vecSize );
	// cleaning
	delete[] vecX;
	delete[] vecY;
	delete[] vecZ;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
