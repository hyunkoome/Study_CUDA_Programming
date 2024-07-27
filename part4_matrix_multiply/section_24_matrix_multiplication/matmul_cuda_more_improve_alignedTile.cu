#include "./common.cpp"

// input parameters
const unsigned TILE_WIDTH = 32;
unsigned matsize = 1024; // num rows and also num cols

// CUDA kernel function: ASSUMPTION: (matsize % TILE_WIDTH == 0)
__global__ void kernelMatMul( float* C, const float* A, const float* B, unsigned matsize, size_t pitch_in_elem ) {
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	register unsigned ntiles = matsize / TILE_WIDTH; // A행렬의 경우 옆으로 가면서 읽을껀데, 타일단위로 끊을것임. 이때, 타일 개수 계산하기 위해서,
	register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y; // y-coord, 저장할때 활용
	register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x; // x-coord, 저장할때 활용
	register float sum = 0.0f;
	for (register unsigned tile = 0; tile < ntiles; ++tile) {
		register unsigned idxA = gy * pitch_in_elem + (tile * TILE_WIDTH + threadIdx.x);
		s_A[threadIdx.y][threadIdx.x] = A[idxA]; // A를 위한 타일, 옆으로 한줄 읽어옴(속도 높이기 위해)
		register unsigned idxB = (tile * TILE_WIDTH + threadIdx.y) * pitch_in_elem + gx;
		s_B[threadIdx.y][threadIdx.x] = B[idxB]; // B를 위한 타일, 옆으로 한줄 읽어옴(속도 높이기 위해)
		__syncthreads();
		for (register unsigned k = 0; k < TILE_WIDTH; ++k) {
			sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x]; // s_: 쉐어드 메모리에 대해서 곱하기를 함을 의미
		}
		__syncthreads();
	}
	register unsigned idxC = gy * pitch_in_elem + gx;
	C[idxC] = sum;
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
	if (matsize % 32 != 0) {
		printf("%s: only accepts multiples of 32\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
	}
	// host-side data
	float* matA = nullptr;
	float* matB = nullptr;
	float* matC = nullptr;
	float* matD = nullptr;
	try {
		matA = new float[matsize * matsize];
		matB = new float[matsize * matsize];
		matC = new float[matsize * matsize];
		matD = new float[matsize * matsize];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, matsize * matsize );
	setNormalizedRandomData( matB, matsize * matsize );
	// device-side data
	float* dev_matA = nullptr;
	float* dev_matB = nullptr;
	float* dev_matC = nullptr;
	// allocate device memory
	ELAPSED_TIME_BEGIN(1);
	size_t host_pitch = matsize * sizeof(float); // host side: packed compactly
	size_t dev_pitch = 0;
	cudaMallocPitch( (void**)&dev_matA, &dev_pitch, matsize * sizeof(float), matsize );
	cudaMallocPitch( (void**)&dev_matB, &dev_pitch, matsize * sizeof(float), matsize );
	cudaMallocPitch( (void**)&dev_matC, &dev_pitch, matsize * sizeof(float), matsize );
	printf("dev_pitch = %zd byte, host_pitch = %zd byte\n", dev_pitch, host_pitch);
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy2D( dev_matA, dev_pitch, matA, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
	cudaMemcpy2D( dev_matB, dev_pitch, matB, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid(div_up(matsize, dimBlock.x), div_up(matsize, dimBlock.y), 1);
	assert(matsize % TILE_WIDTH == 0);
	assert(dev_pitch % sizeof(float) == 0);
	register unsigned pitch_in_elem = dev_pitch / sizeof(float);
	CUDA_PRINT_CONFIG_2D( matsize, matsize );
	ELAPSED_TIME_BEGIN(0);
	kernelMatMul <<< dimGrid, dimBlock>>>( dev_matC, dev_matA, dev_matB, matsize, pitch_in_elem );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy2D( matC, host_pitch, dev_matC, dev_pitch, matsize * sizeof(float), matsize, cudaMemcpyDeviceToHost);
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// free device memory
	cudaFree( dev_matA );
	cudaFree( dev_matB );
	cudaFree( dev_matC );
	CUDA_CHECK_ERROR();
	// check the result
	float sumA = getSum( matA, matsize * matsize );
	float sumB = getSum( matB, matsize * matsize );
	float sumC = getSum( matC, matsize * matsize );
	printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printMat( "matC", matC, matsize, matsize );
	printMat( "matA", matA, matsize, matsize );
	printMat( "matB", matB, matsize, matsize );
#if 0
	memset( matD, 0, matsize * matsize * sizeof(float) );
	for (register unsigned k = 0; k < matsize; ++k) {
		for (register unsigned r = 0; r < matsize; ++r) {
			for (register unsigned c = 0; c < matsize; ++c) {
				unsigned indC = r * matsize + c; // convert to 1D index
				unsigned indA = r * matsize + k; // convert to 1D index
				unsigned indB = k * matsize + c; // convert to 1D index
				matD[indC] += matA[indA] * matB[indB];
			}
		}
	}
	float sumD = getSum( matD, matsize * matsize );
	float rmsError = getRMS( matD, matC, matsize * matsize );
	printf("sumD = %f\n", sumD);
	printf("rmsErr(matC, matD) = %f\n", rmsError);
#endif
	// cleaning
	delete[] matA;
	delete[] matB;
	delete[] matC;
	delete[] matD;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
