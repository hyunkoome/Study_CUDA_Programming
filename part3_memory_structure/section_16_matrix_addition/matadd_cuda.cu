#include "./common.cpp"

// input parameters
unsigned nrow = 10000; // num rows
unsigned ncol = 10000; // num columns

// CUDA kernel function
__global__ void kernel_matadd( float* c, const float* a, const float* b, unsigned nrow, unsigned ncol ) {
	unsigned col = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	unsigned row = blockIdx.y * blockDim.y + threadIdx.y; // CUDA-provided index
	if (row < nrow && col < ncol) {
		unsigned i = row * ncol + col; // converted to 1D index
		c[i] = a[i] + b[i];
	}
}

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		nrow = ncol = procArg( argv[0], argv[1], 32, 16 * 1024 );
		break;
	case 3:
		nrow = procArg( argv[0], argv[1], 32, 16 * 1024 );
		ncol = procArg( argv[0], argv[2], 32, 16 * 1024 );
		break;
	default:
		printf("usage: %s [nrow] [ncol]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// host-side data
	float* matA = nullptr;
	float* matB = nullptr;
	float* matC = nullptr;
	try {
		matA = new float[nrow * ncol];
		matB = new float[nrow * ncol];
		matC = new float[nrow * ncol];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, nrow * ncol );
	setNormalizedRandomData( matB, nrow * ncol );
	// device-side data
	float* dev_matA = nullptr;
	float* dev_matB = nullptr;
	float* dev_matC = nullptr;
	// allocate device memory
	ELAPSED_TIME_BEGIN(1);
	cudaMalloc( (void**)&dev_matA, nrow * ncol * sizeof(float) );
	cudaMalloc( (void**)&dev_matB, nrow * ncol * sizeof(float) );
	cudaMalloc( (void**)&dev_matC, nrow * ncol * sizeof(float) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy( dev_matA, matA, nrow * ncol * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_matB, matB, nrow * ncol * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, (nrow + dimBlock.y - 1) / dimBlock.y, 1);
	CUDA_PRINT_CONFIG_2D( ncol, nrow );
	ELAPSED_TIME_BEGIN(0);
	kernel_matadd <<< dimGrid, dimBlock>>>( dev_matC, dev_matA, dev_matB, nrow, ncol );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( matC, dev_matC, nrow * ncol * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree( dev_matA );
	cudaFree( dev_matB );
	cudaFree( dev_matC );
	CUDA_CHECK_ERROR();
	// check the result
	float sumA = getSum( matA, nrow * ncol );
	float sumB = getSum( matB, nrow * ncol );
	float sumC = getSum( matC, nrow * ncol );
	float diff = fabsf( sumC - (sumA + sumB) );
	printf("matrix size = nrow * ncol = %d * %d\n", nrow, ncol);
	printf("sumC = %f\n", sumC);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / (nrow * ncol) =  %f\n", diff / (nrow * ncol));
	printMat( "matC", matC, nrow, ncol );
	printMat( "matA", matA, nrow, ncol );
	printMat( "matB", matB, nrow, ncol );
	// cleaning
	delete[] matA;
	delete[] matB;
	delete[] matC;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
