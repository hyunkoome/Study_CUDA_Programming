#include "./common.cpp"
#define _USE_MATH_DEFINES // to use M_PI, 프로그램 내에, math.h의 M_PI를 사용하는 것이 가능하도록 하기 위해서.
#include <math.h>

// input parameters
unsigned num = 1024 * 1024; // num of samplings

// host function
void hostMathTest( float* ans, unsigned num ) {
	float inc = 2.0f * (float)(M_PI) / (float)(num);
	float sum = 0.0f;
	for (unsigned i = 0; i < num; ++i) {
		float angle = i * inc;
		sum += sinf(angle) * sinf(angle) + cosf(angle) * cosf(angle);
	}
	ans[0] = sum;
}

// CUDA kernel function
__global__ void kernelMathTest( float* ans, unsigned num ) {
	float inc = 2.0f * (float)(M_PI) / (float)(num);
	float sum = 0.0f;
	for (unsigned i = 0; i < num; ++i) {
		float angle = i * inc;
		sum += sinf(angle) * sinf(angle) + cosf(angle) * cosf(angle);
	}
	ans[1] = sum;
}

// CUDA kernel function
__global__ void kernelFastMathTest( float* ans, unsigned num ) {
	float inc = 2.0f * (float)(M_PI) / (float)(num);
	float sum = 0.0f;
	for (unsigned i = 0; i < num; ++i) {
		float angle = i * inc;
		sum += __sinf(angle) * __sinf(angle) + __cosf(angle) * __cosf(angle);
	}
	ans[2] = sum;
}

// CUDA kernel function
__global__ void kernelFastSinCos( float* ans, unsigned num ) {
	float inc = 2.0f * (float)(M_PI) / (float)(num);
	float sum = 0.0f;
	for (unsigned i = 0; i < num; ++i) {
		float angle = i * inc;
		float sinval;
		float cosval;
		__sincosf(angle, &sinval, &cosval);
		sum += sinval * sinval + cosval * cosval;
	}
	ans[3] = sum;
}

// CUDA kernel function
__global__ void kernelFastSinCosFma( float* ans, unsigned num ) {
	float inc = 2.0f * (float)(M_PI) / (float)(num);
	float sum = 0.0f;
	for (unsigned i = 0; i < num; ++i) {
		float angle = i * inc;
		float sinval;
		float cosval;
		__sincosf(angle, &sinval, &cosval);
		sum = __fmaf_rn(sinval, sinval, sum);
		sum = __fmaf_rn(cosval, cosval, sum);
	}
	ans[4] = sum;
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
	printf("num=%d\n", num);
	// host-side data
	float* ans = new float[5];
	// host kernel call
	printf("host math: ");
	ELAPSED_TIME_BEGIN(0);
	hostMathTest( ans, num );
	ELAPSED_TIME_END(0);
	// device-side data
	float* dev_ans = nullptr;
	// allocate device memory
	cudaMalloc( (void**)&dev_ans, 5 * sizeof(float) );
	cudaMemcpy( dev_ans, ans, 5 * sizeof(float), cudaMemcpyHostToDevice ); // caution: ans[0] contains a result!
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	printf("cuda math: ");
	ELAPSED_TIME_BEGIN(1);
	kernelMathTest <<< 1, 1>>>( dev_ans, num );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch, again
	printf("fast math: ");
	ELAPSED_TIME_BEGIN(2);
	kernelFastMathTest <<< 1, 1>>>( dev_ans, num );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(2);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch, again
	printf("sincos   : ");
	ELAPSED_TIME_BEGIN(3);
	kernelFastSinCos <<< 1, 1>>>( dev_ans, num );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(3);
	CUDA_CHECK_ERROR();
	// CUDA kernel launch, again
	printf("sincosfma: ");
	ELAPSED_TIME_BEGIN(4);
	kernelFastSinCosFma <<< 1, 1>>>( dev_ans, num );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(4);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( ans, dev_ans, 5 * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	// check the result
	printf("host math:\tresult=%f\n", ans[0]);
	printf("cuda math:\tresult=%f\n", ans[1]);
	printf("fast math:\tresult=%f\n", ans[2]);
	printf("sincos   :\tresult=%f\n", ans[3]);
	printf("sincosfma:\tresult=%f\n", ans[4]);
	// free device memory
	cudaFree( dev_ans );
	CUDA_CHECK_ERROR();
	// cleaning
	delete[] ans;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
