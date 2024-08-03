#include "./common.cpp"

// input parameters
const unsigned BLOCK_SIZE = 1024;
unsigned matsize = 16 * 1024; // num rows and also num cols
const float alpha = 0.012;
const float beta = 0.034;

#if defined(_MSC_VER) // CAUTOIN: Visual Studio Compilers need one more definitions for device constants.
__device__ const float dev_alpha = 0.012;
__device__ const float dev_beta = 0.034;
#endif

// host-side data
float* vecZ = nullptr;
float* matA = nullptr;
float* vecX = nullptr;
float* vecY = nullptr;
float* vecW = nullptr;

// device-side data
float* dev_vecZ = nullptr;
float* dev_matA = nullptr;
float* dev_matAtr = nullptr;
float* dev_vecX = nullptr;
float* dev_vecY = nullptr;

// pitch values
size_t host_pitch = matsize * sizeof(float); // host side: packed compactly
size_t dev_pitch = 0;
size_t pitch_in_elem = 0;

// CUDA kernel function
__global__ void kernelMatTranspose( float* C, const float* A, unsigned matsize, size_t pitch_in_elem ) {
	__shared__ float mat[32][32 + 1];
	// pick up for the shared memory
	register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y; // CUDA-provided index
	register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (gy < matsize && gx < matsize) {
		register unsigned idxA = gy * pitch_in_elem + gx;
		mat[threadIdx.y][threadIdx.x] = A[idxA];
#if defined(_DEBUG)
	} else {
		mat[threadIdx.y][threadIdx.x] = -1.0f; // if you got a "-1.0" in your result, it was out of bound.
#endif
	}
	__syncthreads();
	// transposed position
	gy = blockIdx.x * blockDim.x + threadIdx.y; // CUDA-provided index
	gx = blockIdx.y * blockDim.y + threadIdx.x; // CUDA-provided index
	if (gy < matsize && gx < matsize) {
		register unsigned idxC = gy * pitch_in_elem + gx;
		C[idxC] = mat[threadIdx.x][threadIdx.y];
	}
}

// CUDA kernel function
__global__ void kernelGEMVtransposed( float* Z, const float* Atr, const float* X, const float* Y,
                                      unsigned matsize, size_t pitch_in_elem ) {
	__shared__ float s_X[BLOCK_SIZE];
	register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y; // CUDA-provided index
	if (gy < matsize) {
		register float sum = 0.0f;
		register unsigned remaining = matsize;
		register unsigned ntiles = div_up(matsize, BLOCK_SIZE);
		for (register unsigned tile = 0; tile < ntiles; ++tile) {
			register unsigned nelem = min( remaining,  BLOCK_SIZE );
			remaining -= BLOCK_SIZE;
			if (threadIdx.y < nelem) {
				register unsigned index = tile * BLOCK_SIZE + threadIdx.y;
				s_X[threadIdx.y] = X[index];
			}
			__syncthreads();
			register unsigned idxA = (tile * BLOCK_SIZE) * pitch_in_elem + gy;
			const float* ptr = Atr + idxA;
			for (register unsigned k = 0; k < nelem; ++k) {
				sum += *ptr * s_X[k];
				ptr += pitch_in_elem;
			}
			__syncthreads();
		}
#if ! defined(_MSC_VER) // GNU compilers can handle C++ constants properly, even in CUDA kernels.
		Z[gy] = alpha * sum + beta * Y[gy];
#else // CAUTOIN: Visual Studio Compilers need one more definitions for device constants.
		Z[gy] = dev_alpha * sum + dev_beta * Y[gy];
#endif
	}
}

// execute
void execute(void) {
	ELAPSED_TIME_BEGIN(1);
	// H2D copy
	cudaMemcpy2D( dev_matA, dev_pitch, matA, host_pitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_vecX, vecX, matsize * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_vecY, vecY, matsize * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// transpose matrix
	const unsigned blocksize = 32; // because 32 * 32 = 1024
	dim3 blockConf(blocksize, blocksize, 1);
	dim3 gridConf(div_up(matsize, blockConf.x), div_up(matsize, blockConf.y), 1);
	assert(dev_pitch % sizeof(float) == 0);
	pitch_in_elem = dev_pitch / sizeof(float);
	ELAPSED_TIME_BEGIN(2);
	kernelMatTranspose <<< gridConf, blockConf>>>( dev_matAtr, dev_matA, matsize, pitch_in_elem );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(2);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(1, BLOCK_SIZE, 1);
	dim3 dimGrid(1, div_up(matsize, dimBlock.y), 1);
	CUDA_PRINT_CONFIG( matsize );
	ELAPSED_TIME_BEGIN(0);
	kernelGEMVtransposed <<< dimGrid, dimBlock>>>( dev_vecZ, dev_matAtr, dev_vecX, dev_vecY, matsize, pitch_in_elem );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// D2H copy
	cudaMemcpy( vecZ, dev_vecZ, matsize * sizeof(float), cudaMemcpyDeviceToHost );
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
}

// check
void check(void) {
	// check the result
	for (register unsigned y = 0; y < matsize; ++y) {
		register float ans_ax = 0.0f;
		for (register unsigned k = 0; k < matsize; ++k) {
			unsigned indA = y * matsize + k; // convert to 1D index
			ans_ax += matA[indA] * vecX[k];
		}
		vecW[y] = alpha * ans_ax + beta * vecY[y];
	}
	float sumZ = getSum( vecZ, matsize );
	float sumA = getSum( matA, matsize * matsize );
	float sumX = getSum( vecX, matsize );
	float sumY = getSum( vecY, matsize );
	printf("sumZ = %f\n", sumZ);
	printf("sumA = %f\n", sumA);
	printf("sumX = %f\n", sumX);
	printf("sumY = %f\n", sumY);
	printVec( "vecZ", vecZ, matsize );
	printMat( "matA", matA, matsize, matsize );
	printVec( "vecX", vecX, matsize );
	printVec( "vecY", vecY, matsize );
	float sumW = getSum( vecW, matsize );
	float rmsError = getRMS( vecZ, vecW, matsize );
	printf("sumW = %f\n", sumW);
	printf("rmsErr(vecZ, vecW) = %f\n", rmsError);
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
	printf("PROBLEM SIZE: matsize = %d * %d\n", matsize, matsize);
	// host-side alloc
	try {
		vecZ = new float[matsize];
		matA = new float[matsize * matsize];
		vecX = new float[matsize];
		vecY = new float[matsize];
		vecW = new float[matsize];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	host_pitch = matsize * sizeof(float); // host side: packed compactly
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, matsize * matsize );
	setNormalizedRandomData( vecX, matsize );
	setNormalizedRandomData( vecY, matsize );
	// allocate device memory
	cudaMalloc( (void**)&dev_vecZ, matsize * sizeof(float) );
	cudaMallocPitch( (void**)&dev_matA, &dev_pitch, matsize * sizeof(float), matsize );
	cudaMallocPitch( (void**)&dev_matAtr, &dev_pitch, matsize * sizeof(float), matsize );
	cudaMalloc( (void**)&dev_vecX, matsize * sizeof(float) );
	cudaMalloc( (void**)&dev_vecY, matsize * sizeof(float) );
	printf("dev_pitch = %zd byte, host_pitch = %zd byte\n", dev_pitch, host_pitch);
	CUDA_CHECK_ERROR();
	// execute the kernel twice
	execute();
	check();
	// free device memory
	cudaFree( dev_vecZ );
	cudaFree( dev_matA );
	cudaFree( dev_matAtr );
	cudaFree( dev_vecX );
	cudaFree( dev_vecY );
	// cleaning
	delete[] vecZ;
	delete[] matA;
	delete[] vecX;
	delete[] vecY;
	delete[] vecW;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
