#include "./common.cpp"

const unsigned SIZE = 1024 * 1024; // 1M elements

// CUDA kernel function
__global__ void singleKernelVecAdd( float* c, const float* a, const float* b ) {
	for (register unsigned i = 0; i < SIZE; ++i) {
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
    // single core로 1개의 core만 사용하는 걸로..
    // 아마도, gpu core 1개는 cpu보다 느리니, cpu core 사용한거보다 느리게 나올듯.. 확인해보자!!
    // 확인해보면 cpu 1개로 돌리는 것 보다, CUDA 1개로 돌리는게 엄청~ 느려짐
	ELAPSED_TIME_BEGIN(0);
	singleKernelVecAdd <<< 1, 1>>>( dev_vecC, dev_vecA, dev_vecB );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0); // cuda 메모리 카피 제외된 시간 측정, 즉, 진짜 커널만 돌리는 시간 측정
	CUDA_CHECK_ERROR();

	// copy to host from device
	cudaMemcpy( vecC, dev_vecC, SIZE * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1); // cuda 메모리 카피 포함된 시간 측정, 메모리 카피(즉, CPU MEM <-> GPU VRAM 카피 시간이 오래걸림)
    // 실제로 CUDA 프로그램 돌리려면, 커널만 돌리는게 아니라, 메인 메모리에서 VRAM 으로 카피하는 것도 필요하므로,
    // 그래서, 카피하는 시간, 역으로 카피 해오는 시간도 포함해서, 측정 함

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
