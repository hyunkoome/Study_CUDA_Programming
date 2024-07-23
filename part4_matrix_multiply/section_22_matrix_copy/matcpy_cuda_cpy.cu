#include "./common.cpp"

// input parameters
unsigned matsize = 4000; // num rows and also num cols

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
	ELAPSED_TIME_BEGIN(0);
	cudaMemcpy2D( dev_matC, dev_pitch, dev_matA, dev_pitch, matsize * sizeof(float), matsize, cudaMemcpyDeviceToDevice);
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
