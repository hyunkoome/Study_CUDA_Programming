#include "./common.cpp"

// input parameters
unsigned vecSize = 256 * 1024 * 1024; // 256M elements

// CUDA kernel function
__global__ void kernelVecAdd( float* c, const float* a, const float* b, unsigned n, long long* times ) {
	clock_t start = clock();
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < n) {
		c[i] = a[i] + b[i];
	}
	clock_t end = clock();
	if (i == 0) {
		times[0] = (long long)(end - start);
	}
}


int main( const int argc, const char* argv[] ) {
	// argv processing
	char* pEnd = nullptr;
	switch (argc) {
	case 1:
        // 파라미터 값이 없으면,
        // default 값인 global 변수인 vecSize의 값을 그냥 사용
		break;
	case 2:
        // strtol: string to long: 문자열을 long 타입의 숫자로 바꿈
        // 이 값을 global 변수인 vecSize에 넣음
		vecSize = strtol( argv[1], &pEnd, 10 );
		break;
	default:
		printf("usage: %s [size]\n", argv[0]);
		exit( EXIT_FAILURE );
		break;
	}
	if (vecSize < 1) {
		printf("%s: ERROR: invalid size = %d\n", argv[0], vecSize);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
	}
	// host-side data
	float* vecA = new float[vecSize];
	float* vecB = new float[vecSize];
	float* vecC = new float[vecSize];
	long long* host_times = new long long[1];
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecA, vecSize );
	setNormalizedRandomData( vecB, vecSize );
	// device-side data
	float* dev_vecA = nullptr;
	float* dev_vecB = nullptr;
	float* dev_vecC = nullptr;
	long long* dev_times = nullptr;
	// allocate device memory
	cudaMalloc( (void**)&dev_vecA, vecSize * sizeof(float) );
	cudaMalloc( (void**)&dev_vecB, vecSize * sizeof(float) );
	cudaMalloc( (void**)&dev_vecC, vecSize * sizeof(float) );
	cudaMalloc( (void**)&dev_times, 1 * sizeof(long long) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy( dev_vecA, vecA, vecSize * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_vecB, vecB, vecSize * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock( 1024, 1, 1 );
	dim3 dimGrid( (vecSize + (dimBlock.x - 1)) / dimBlock.x, 1, 1 );
	CUDA_PRINT_CONFIG( vecSize );
	ELAPSED_TIME_BEGIN(0);
	kernelVecAdd <<< dimGrid, dimBlock>>>( dev_vecC, dev_vecA, dev_vecB, vecSize, dev_times );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( vecC, dev_vecC, vecSize * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( host_times, dev_times, 1 * sizeof(long long), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree( dev_vecA );
	cudaFree( dev_vecB );
	cudaFree( dev_vecC );
	cudaFree( dev_times );
	CUDA_CHECK_ERROR();
	// kernel clock calculation
	int peak_clk = 1;
	cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
	printf("num clock = %lld, peak clock rate = %dkHz, elapsed time: %f usec\n",
	       host_times[0], peak_clk, host_times[0] * 1000.0f / (float)peak_clk);
	// check the result
	float sumA = getSum( vecA, vecSize );
	float sumB = getSum( vecB, vecSize );
	float sumC = getSum( vecC, vecSize );
	float diff = fabsf( sumC - (sumA + sumB) );
	printf("SIZE = %d\n", vecSize);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / SIZE =  %f\n", diff / vecSize);
	printVec( "vecA", vecA, vecSize );
	printVec( "vecB", vecB, vecSize );
	printVec( "vecC", vecC, vecSize );
	// cleaning
	delete[] vecA;
	delete[] vecB;
	delete[] vecC;
	delete[] host_times;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
