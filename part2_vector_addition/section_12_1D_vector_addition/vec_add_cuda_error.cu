#include "./common.cpp"

const unsigned SIZE = 1024 * 1024; // 1M elements

// CUDA kernel function
__global__ void kernelVecAdd( float* c, const float* a, const float* b, unsigned n ) {
	unsigned i = threadIdx.x; // CUDA-provided index

    // n 으로 배열의 크기(SIZE)를 줌
    // 경우에 따라서는, thread 개수보다, 배열의 크기가 더 많을 경우가 있음.
    // 또는, thread index를 구했는데, 내가 배열의 바깥 부분을 건드리는 것은 아닌지, double 체크 하려고.
    // 그래서, thread index 가 배열을 안 벗어나는 경우에만 실행하려고 if (i<n) 으로 감쌈
    // thread index 잘못 계산하는 경우에, 아예 실행이 안되서, 메모리를 침범하는 일을 막아둠.
	if (i < n) {
		c[i] = a[i] + b[i]; // 딱 1번 돌림
	}
}

// 돌리면 error 뜨는 게 맞음


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

    // SIZE  = 100만개(1024 x 1024) core를 사용 요구 해서 돌림
    // error 뜨는 이유: 우리 GPU에 100만게 core가 없음.
    // 그런데, 100만개의 코어를 요구해서 돌리려고 함 => 안되죠 !
    // 그래서, 안되는 것을 한번, 확인해보자!
    // cuda failure "invalid configuration argument" : CUDA 커널을 만들때 이 커널이 잘못 configure 됬다는 의미임
    // SM (streaming multi-processor)에서 1M개의 thread를 동시 실행 불가능 -> 실제로는 1024개가 한계
	kernelVecAdd <<< 1, SIZE>>>( dev_vecC, dev_vecA, dev_vecB, SIZE );
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
