#include "./common.cpp"

// input parameters
const float alpha = 0.5f;
const float beta = -100.0f;
const unsigned TILE_WIDTH = 32;
unsigned matsize = 4096; // num rows and also num cols

// CUDA kernel function
__global__ void kernelGEMM( float* Z, const float* A, const float* B, const float* C,
                            unsigned matsize, size_t pitch_in_elem, const float alpha, const float beta ) {
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	register unsigned ntiles = (matsize + TILE_WIDTH - 1) / TILE_WIDTH;
	register unsigned remaining = matsize; // remained elements to be multiplied
	register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y; // y-coord
	register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x; // x-coord
	register float sum = 0.0f;
	for (register unsigned tile = 0; tile < ntiles; ++tile) {
		register unsigned nelem = min( remaining, TILE_WIDTH );
		remaining -= TILE_WIDTH;

        // 속도 개선을 위해 앞쪽 tile 에서는 if 문 없이 막 돌릴고, 뒤쪽 tile 에서는 if문 돌리는 경우도 있음
        // (gemm_cuda_tile_optim.cu 참고)
		if (gy < matsize && threadIdx.x < nelem) {
			register unsigned idxA = gy * pitch_in_elem + (tile * TILE_WIDTH + threadIdx.x);
			s_A[threadIdx.y][threadIdx.x] = A[idxA];
#if defined(_DEBUG)
		} else {
			s_A[threadIdx.y][threadIdx.x] = 0.0f;
#endif
		}
		if (gx < matsize && threadIdx.y < nelem) {
			register unsigned idxB = (tile * TILE_WIDTH + threadIdx.y) * pitch_in_elem + gx;
			s_B[threadIdx.y][threadIdx.x] = B[idxB];
#if defined(_DEBUG)
		} else {
			s_B[threadIdx.y][threadIdx.x] = 0.0f;
#endif
		}
		__syncthreads();
		for (register unsigned k = 0; k < nelem; ++k) {
			sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
		}
		__syncthreads();
	}
	if (gy < matsize && gx < matsize) {
		register unsigned idxC = gy * pitch_in_elem + gx;
		Z[idxC] = alpha * sum + beta * C[idxC];
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
	float* matB = nullptr;
	float* matC = nullptr;
	float* matZ = nullptr;
	float* matW = nullptr;
	try {
		matA = new float[matsize * matsize];
		matB = new float[matsize * matsize];
		matC = new float[matsize * matsize];
		matZ = new float[matsize * matsize];
		matW = new float[matsize * matsize];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, matsize * matsize );
	setNormalizedRandomData( matB, matsize * matsize );
	setNormalizedRandomData( matC, matsize * matsize );
	// device-side data
	float* dev_matA = nullptr;
	float* dev_matB = nullptr;
	float* dev_matC = nullptr;
	float* dev_matZ = nullptr;
	// allocate device memory
	ELAPSED_TIME_BEGIN(1);
	size_t host_pitch = matsize * sizeof(float); // host side: packed compactly
	size_t dev_pitch = 0;
	cudaMallocPitch( (void**)&dev_matA, &dev_pitch, matsize * sizeof(float), matsize );
	cudaMallocPitch( (void**)&dev_matB, &dev_pitch, matsize * sizeof(float), matsize );
	cudaMallocPitch( (void**)&dev_matC, &dev_pitch, matsize * sizeof(float), matsize );
	cudaMallocPitch( (void**)&dev_matZ, &dev_pitch, matsize * sizeof(float), matsize );
	printf("dev_pitch = %zd byte, host_pitch = %zd byte\n", dev_pitch, host_pitch);
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy2D( dev_matA, dev_pitch, matA, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
	cudaMemcpy2D( dev_matB, dev_pitch, matB, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
	cudaMemcpy2D( dev_matC, dev_pitch, matC, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(div_up(matsize, dimBlock.x), div_up(matsize, dimBlock.y), 1);
	assert(dev_pitch % sizeof(float) == 0);
	register unsigned pitch_in_elem = dev_pitch / sizeof(float);
	CUDA_PRINT_CONFIG_2D( matsize, matsize );
	ELAPSED_TIME_BEGIN(0);
	kernelGEMM <<< dimGrid, dimBlock>>>( dev_matZ, dev_matA, dev_matB, dev_matC, matsize, pitch_in_elem, alpha, beta );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy2D( matZ, host_pitch, dev_matZ, dev_pitch, matsize * sizeof(float), matsize, cudaMemcpyDeviceToHost);
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// free device memory
	cudaFree( dev_matA );
	cudaFree( dev_matB );
	cudaFree( dev_matC );
	cudaFree( dev_matZ );
	CUDA_CHECK_ERROR();
	// check the result
	float sumA = getSum( matA, matsize * matsize );
	float sumB = getSum( matB, matsize * matsize );
	float sumC = getSum( matC, matsize * matsize );
	float sumZ = getSum( matZ, matsize * matsize );
	printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("sumZ = %f\n", sumZ);
	printMat( "matZ", matZ, matsize, matsize );
	printMat( "matA", matA, matsize, matsize );
	printMat( "matB", matB, matsize, matsize );
	printMat( "matC", matC, matsize, matsize );
#if 0
	memset( matW, 0, matsize * matsize * sizeof(float) );
	for (register unsigned k = 0; k < matsize; ++k) {
		for (register unsigned r = 0; r < matsize; ++r) {
			for (register unsigned c = 0; c < matsize; ++c) {
				unsigned indZ = r * matsize + c; // convert to 1D index
				unsigned indA = r * matsize + k; // convert to 1D index
				unsigned indB = k * matsize + c; // convert to 1D index
				matW[indZ] += matA[indA] * matB[indB];
			}
		}
	}
	for (register unsigned r = 0; r < matsize; ++r) {
		for (register unsigned c = 0; c < matsize; ++c) {
			unsigned indW = r * matsize + c; // convert to 1D index
			matW[indW] = alpha * matW[indW] + beta * matC[indW];
		}
	}
	float sumW = getSum( matW, matsize * matsize );
	float rmsError = getRMS( matW, matZ, matsize * matsize );
	printf("sumW = %f\n", sumW);
	printf("rmsErr(matZ, matW) = %f\n", rmsError);
#endif
	// cleaning
	delete[] matA;
	delete[] matB;
	delete[] matC;
	delete[] matZ;
	delete[] matW;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
