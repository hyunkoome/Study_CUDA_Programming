#include "./common.cpp"

const unsigned SIZE = 1024 * 1024; // 1M elements

// CUDA kernel function
__global__ void kernelVecAdd( float* c, const float* a, const float* b, unsigned n ) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}


int main(void) {
	// host-side data
	float* vecA = new float[SIZE];
	float* vecB = new float[SIZE];
	float* vecC = new float[SIZE];

	// set random data
	srand( 0 );
	setNormalizedRandomData( vecA, SIZE );
	setNormalizedRandomData( vecB, SIZE );

	// device-side data
	float* dev_vecA = nullptr;
	float* dev_vecB = nullptr;
	float* dev_vecC = nullptr;

	// allocate device memory
	cudaMalloc( (void**)&dev_vecA, SIZE * sizeof(float) );
	cudaMalloc( (void**)&dev_vecB, SIZE * sizeof(float) );
	cudaMalloc( (void**)&dev_vecC, SIZE * sizeof(float) );
	CUDA_CHECK_ERROR();

	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy( dev_vecA, vecA, SIZE * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_vecB, vecB, SIZE * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();

	// CUDA kernel launch
	ELAPSED_TIME_BEGIN(0);

    // SIZE = 1024x1024, 즉, 100만개
    // blockDim = 1024: 쓰레드 블록에는 1024개를 돌리도록 하고, 즉, 블록 1개 안에는 1024개가 돌아가고
    // gridDim = SIZE / 1024: 블록 개수 계산, 100만개를 이 쓰레드 블럭 1개에 1024개가 돌라가니깐, 1M/1024 나눠버리면 => 블록의 개수가 나옴
	kernelVecAdd <<< SIZE / 1024, 1024>>>( dev_vecC, dev_vecA, dev_vecB, SIZE );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();

	// copy to host from device
	cudaMemcpy( vecC, dev_vecC, SIZE * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);

	// free device memory
	cudaFree( dev_vecA );
	cudaFree( dev_vecB );
	cudaFree( dev_vecC );
	CUDA_CHECK_ERROR();

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
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
