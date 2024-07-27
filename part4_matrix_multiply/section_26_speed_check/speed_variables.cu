#include "./common.cpp"
#define _USE_MATH_DEFINES // to use M_PI
#include <math.h>

// input parameters
const unsigned SHMEM_SIZE = 8 * 1024; // elements
unsigned num = 1024 * 1024; // num of samplings

__constant__ float c_value = 1.000f;

// CUDA kernel function
__global__ void kernelSpeedTest( float* g_a, unsigned num, float* ans, long long* times ) {
	clock_t begin, end;
	int gx = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	// case 0: register variable
	begin = clock();
	register float a = 0.0f;
	register float b = 1.0f;
	for (unsigned i = 0; i < num; ++i) {
		a = a + b;
	}
	end = clock();
	if (gx == 0) {
		ans[0] = a;
		times[0] = (long long)(end - begin);
	}
	// case 1: shared memory variable
	begin = clock();
	__shared__ float s_a[SHMEM_SIZE]; // maximum amount for current machine
	s_a[0] = 0.0f;
	for (unsigned j = 0; j < num / SHMEM_SIZE; ++j) {
		for (unsigned i = 0; i < SHMEM_SIZE; ++i) {
			unsigned next = (i + 1) % SHMEM_SIZE;
			s_a[next] = s_a[i] + 1.0f;
		}
	}
	end = clock();
	if (gx == 0) {
		unsigned ind = num % SHMEM_SIZE;
		ans[1] = s_a[ind];
		times[1] = (long long)(end - begin);
	}
	// case 2: global memory variable
	begin = clock();
	g_a[0] = 0.0f;
	for (unsigned i = 0; i < num; ++i) {
		unsigned next = (i + 1) % num;
		g_a[next] = g_a[i] + 1.0f;
	}
	end = clock();
	if (gx == 0) {
		ans[2] = g_a[0];
		times[2] = (long long)(end - begin);
	}
	// case 3: local memory variable
	begin = clock();
	float l_a[SHMEM_SIZE];
	l_a[0] = 0.0f;
	for (unsigned j = 0; j < num / SHMEM_SIZE; ++j) {
		for (unsigned i = 0; i < SHMEM_SIZE; ++i) {
			unsigned next = (i + 1) % SHMEM_SIZE;
			l_a[next] = l_a[i] + 1.0f;
		}
	}
	end = clock();
	if (gx == 0) {
		unsigned ind = num % SHMEM_SIZE;
		ans[3] = l_a[ind];
		times[3] = (long long)(end - begin);
	}
	// case 4: register + const variable
	begin = clock();
	register float c = 0.0f;
	for (unsigned i = 0; i < num; ++i) {
		c = c + c_value;
	}
	end = clock();
	if (gx == 0) {
		ans[4] = c;
		times[4] = (long long)(end - begin);
	}
}

int main( const int argc, const char* argv[] ) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1 );
		break;
	default:
		printf("usage: %s [num]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	if (num % SHMEM_SIZE != 0) {
		printf("CAUTION: num=%u is NOT a multiple of SHMEM_SIZE=%d.\n", num, SHMEM_SIZE);
	}
	printf("num=%u, SHMEM_SIZE=%u\n", num, SHMEM_SIZE);
	// host-side data
	float* ans = new float[6];
	long long* times = new long long[6];
	// device-side data
	float* dev_g_a = nullptr;
	float* dev_ans = nullptr;
	long long* dev_times = nullptr;
	// allocate device memory
	cudaMalloc( (float**)&dev_g_a, num * sizeof(float) );
	cudaMalloc( (float**)&dev_ans, 6 * sizeof(float) );
	cudaMalloc( (long long**)&dev_times, 6 * sizeof(long long) );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	ELAPSED_TIME_BEGIN(0);
	kernelSpeedTest <<< 1, 1>>>( dev_g_a, num, dev_ans, dev_times );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( ans, dev_ans, 6 * sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy( times, dev_times, 6 * sizeof(long long), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	// check the result
	printf("reg var case:   \tans=%f\tclock=%12lld ticks\n", ans[0], times[0]);
	printf("shared var case:\tans=%f\tclock=%12lld ticks\n", ans[1], times[1]);
	printf("global var case:\tans=%f\tclock=%12lld ticks\n", ans[2], times[2]);
	printf("local var case: \tans=%f\tclock=%12lld ticks\n", ans[3], times[3]);
	printf("const var case: \tans=%f\tclock=%12lld ticks\n", ans[4], times[4]);
	// free device memory
	cudaFree( dev_g_a );
	cudaFree( dev_ans );
	cudaFree( dev_times );
	CUDA_CHECK_ERROR();
	// cleaning
	delete[] ans;
	delete[] times;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
