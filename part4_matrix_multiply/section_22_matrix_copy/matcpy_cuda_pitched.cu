#include "./common.cpp"

// input parameters
unsigned matsize = 4000; // num rows and also num cols

// CUDA kernel function
__global__ void kernelMatCpy( float* C, const float* A, int matsize, size_t pitch_in_elem ) {
	register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y; // CUDA-provided index
	if (gy < matsize) {
		register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
		if (gx < matsize) {
			register unsigned idx = gy * pitch_in_elem + gx; // in element
			C[idx] = A[idx];
		}
	}
}

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		matsize = procArg( argv[0], argv[1], 4 );
		break;
	default:
		printf("usage: %s [matsize]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// host-side data
	float* matA = nullptr;
	float* matC = nullptr;
	try {
		matA = new float[matsize * matsize];
		matC = new float[matsize * matsize];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, matsize * matsize );
	// device-side data
	float* dev_matA = nullptr;
	float* dev_matC = nullptr;
	// allocate device memory
	ELAPSED_TIME_BEGIN(1);
	size_t host_pitch = matsize * sizeof(float); // host side: packed compactly
	size_t dev_pitch = 0;
	cudaMallocPitch( (void**)&dev_matA, &dev_pitch, matsize * sizeof(float), matsize );
	cudaMallocPitch( (void**)&dev_matC, &dev_pitch, matsize * sizeof(float), matsize );
	printf("dev_pitch = %zd byte, host_pitch = %zd byte\n", dev_pitch, host_pitch);
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy2D( dev_matA, dev_pitch, matA, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(div_up(matsize, dimBlock.x), div_up(matsize, dimBlock.y), 1);
	assert(dev_pitch % sizeof(float) == 0);
	register unsigned pitch_in_elem = dev_pitch / sizeof(float);
	CUDA_PRINT_CONFIG_2D( matsize, matsize );
	ELAPSED_TIME_BEGIN(0);
	kernelMatCpy <<< dimGrid, dimBlock>>>( dev_matC, dev_matA, matsize, pitch_in_elem );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy2D( matC, host_pitch, dev_matC, dev_pitch, matsize * sizeof(float), matsize, cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree( dev_matA );
	cudaFree( dev_matC );
	CUDA_CHECK_ERROR();
	// check the result
	float sumA = getSum( matA, matsize * matsize );
	float sumC = getSum( matC, matsize * matsize );
	float diff = fabsf( sumC - sumA );
	printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
	printf("sumA = %f\n", sumA);
	printf("sumC = %f\n", sumC);
	printf("diff(sumA, sumC) = %f\n", diff);
	printf("diff(sumA, sumC) / SIZE = %f\n", diff / (matsize * matsize));
	printMat( "matA", matA, matsize, matsize );
	printMat( "matC", matC, matsize, matsize );
	// cleaning
	delete[] matA;
	delete[] matC;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
