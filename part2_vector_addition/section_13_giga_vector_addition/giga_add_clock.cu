#include "./common.cpp"

const unsigned SIZE = 256 * 1024 * 1024; // 256 elements

// CUDA kernel function
__global__ void kernelVecAdd( float* c, const float* a, const float* b, unsigned n, long long* times ) {
    /*
    - clock() in 커널 함수 활용
    - clock_t clock(void);
    - long long int clock64(void); `64비트`
    - returns the value of a per-multiprocessor counter (or clock ticks)
    - CAUTION: executed in __device__ and __global__ functions
    - 실제로는, 앞에 __device__ 나 __global__ 이 붙은 쿠다 커널 함수 또는 CUDA의 디바이스 함수 내에서 돌릴 수 있음
    - 즉, cpu 가 아니라, CUDA 내에서 clock tick 이 얼마나 되었는지 알려줌
    - 그래서, 몇 클럭 만에 수행됬는지, 클럭 획수를 알려주는 식으로 되어 있음
    - 그런데, 이것을 메인함수에 어떻게 알려주느냐?
    - 그래서, 64bit 정수를 저장할수 있는 배열(디바이스에 있는 배열)을 하나 더 주고,
    - 커널 함수 파라미터로 long long* times 를 추가..
    - i 번째 실행시간을 times[i]에 클럭 단위로 넣도록 이렇게 구현
    */
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


int main(void) {
	// host-side data
	float* vecA = new float[SIZE];
	float* vecB = new float[SIZE];
	float* vecC = new float[SIZE];
	long long* host_times = new long long[1];
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecA, SIZE );
	setNormalizedRandomData( vecB, SIZE );
	// device-side data
	float* dev_vecA = nullptr;
	float* dev_vecB = nullptr;
	float* dev_vecC = nullptr;
	long long* dev_times = nullptr; //
	// allocate device memory
	cudaMalloc( (void**)&dev_vecA, SIZE * sizeof(float) );
	cudaMalloc( (void**)&dev_vecB, SIZE * sizeof(float) );
	cudaMalloc( (void**)&dev_vecC, SIZE * sizeof(float) );
	cudaMalloc( (void**)&dev_times, 1 * sizeof(long long) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy( dev_vecA, vecA, SIZE * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_vecB, vecB, SIZE * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock( 1024, 1, 1 );
	dim3 dimGrid( (SIZE + dimBlock.x - 1) / dimBlock.x, 1, 1 );
	CUDA_PRINT_CONFIG( SIZE );
	ELAPSED_TIME_BEGIN(0);
	kernelVecAdd <<< dimGrid, dimBlock>>>( dev_vecC, dev_vecA, dev_vecB, SIZE, dev_times );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( vecC, dev_vecC, SIZE * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( host_times, dev_times, 1 * sizeof(long long), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree( dev_vecA );
	cudaFree( dev_vecB );
	cudaFree( dev_vecC );
	cudaFree( dev_times );
	CUDA_CHECK_ERROR();

    // clock의 실제 수행 시간을 알려면, clock을 수행하는데 필요한 Hz를 알아야 함
    // kernel clock calculation
	int peak_clk = 1;
    // clock rate 를 KHz 단위로 알려줌
    // 마지막 파라미터는 gpu device 번호
	cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
    // micro sec, 즉, 100만분의 1초 단위로 얼마나 걸렸는지를 체크
	printf("num clock = %lld, peak clock rate = %dkHz, elapsed time: %f usec\n",
	       host_times[0], peak_clk, host_times[0] * 1000.0f / (float)peak_clk);
	// check the result
	float sumA = getSum( vecA, SIZE );
	float sumB = getSum( vecB, SIZE );
	float sumC = getSum( vecC, SIZE );
	float diff = fabsf( sumC - (sumA + sumB) );
	printf("SIZE = %d\n", SIZE);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / SIZE =  %f\n", diff / SIZE);
	printVec( "vecA", vecA, SIZE );
	printVec( "vecB", vecB, SIZE );
	printVec( "vecC", vecC, SIZE );
	// cleaning
	delete[] vecA;
	delete[] vecB;
	delete[] vecC;
	delete[] host_times;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
