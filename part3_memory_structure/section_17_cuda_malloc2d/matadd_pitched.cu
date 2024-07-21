#include "./common.cpp"

// input parameters
unsigned nrow = 10000; // num rows
unsigned ncol = 100000; // num columns

// CUDA kernel function
__global__ void kernel_matadd( float* c, const float* a, const float* b,
                               unsigned nrow, unsigned ncol, size_t dev_pitch ) {
	register unsigned col = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (col < ncol) {
		register unsigned row = blockIdx.y * blockDim.y + threadIdx.y; // CUDA-provided index
		if (row < nrow) {
            // offset을 dev_pitch 단위로 계산해야 함
			register unsigned offset = row * dev_pitch + col * sizeof(float); // in byte
			*((float*)((char*)c + offset)) = *((const float*)((const char*)a + offset))
			                                 + *((const float*)((const char*)b + offset));
		}
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
	size_t host_pitch = ncol * sizeof(float); // host side: packed compactly
	size_t dev_pitch = 0;

    // cudaMallocPitch
	cudaMallocPitch( (void**)&dev_matA, &dev_pitch, ncol * sizeof(float), nrow );
	cudaMallocPitch( (void**)&dev_matB, &dev_pitch, ncol * sizeof(float), nrow );
	cudaMallocPitch( (void**)&dev_matC, &dev_pitch, ncol * sizeof(float), nrow );
	printf("dev_pitch = %zd byte, host_pitch = %zd byte\n", dev_pitch, host_pitch);
	CUDA_CHECK_ERROR();

	// copy to device from host
    // cudaMemcpy2D
	cudaMemcpy2D( dev_matA, dev_pitch, matA, host_pitch, ncol * sizeof(float), nrow, cudaMemcpyHostToDevice);
	cudaMemcpy2D( dev_matB, dev_pitch, matB, host_pitch, ncol * sizeof(float), nrow, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();

	// CUDA kernel launch
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, (nrow + dimBlock.y - 1) / dimBlock.y, 1);
	CUDA_PRINT_CONFIG_2D( ncol, nrow );
	ELAPSED_TIME_BEGIN(0);
	kernel_matadd <<< dimGrid, dimBlock>>>( dev_matC, dev_matA, dev_matB, nrow, ncol, dev_pitch );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();

    // copy to host from device
	cudaMemcpy2D( matC, host_pitch, dev_matC, dev_pitch, ncol * sizeof(float), nrow, cudaMemcpyDeviceToHost);
	chrono::system_clock::time_point timeEnd2 = chrono::system_clock::now();
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
