#include "./common.cpp"

const unsigned SIZE = 256 * 1024 * 1024; // big-size elements

// CUDA kernel function
__global__ void kernelVecAdd( float* c, const float* a, const float* b, unsigned n ) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
    // 256M 개수중 딱 내것만 즉 a[i], b[i] 가져와서, c[i] 에 저장하는 것만 처리 함
    // blockDim 과 vector size가 맞아 떨어진다는 보장이 없음
    // 예, vector size = 999, block 은 32배수로 실행 됨
    // 전체 thread 개수 = dimBlock.x * dumGrid.x >= SIZE 를 보장해야.
    // dimGrid.x = ceil(SIZE / dimBlock.x) = 올림(SIZE / dimBlock.x) = (SIZE + (dimBlock.x -1)) / dimBlock.x
    // 올림(A/B) = (A+(B-1))/B
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}


int main(void) {
	// host-side data
	float* vecA = nullptr;
	float* vecB = nullptr;
	float* vecC = nullptr;
	try {
		vecA = new float[SIZE];
		vecB = new float[SIZE];
		vecC = new float[SIZE];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(1);
	}

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
    //.. dimBlock은 default로 이제 1024로 그냥 적자!
	dim3 dimBlock( 1024, 1, 1 );

    //.. 전체 thread 개수 = dimBlock.x * dumGrid.x >= SIZE 를 보장해야.
    //.. dimGrid.x = ceil(SIZE / dimBlock.x) = 올림(SIZE / dimBlock.x) = (SIZE + (dimBlock.x -1)) / dimBlock.x
    //.. 올림(A/B) = (A+(B-1))/B
	dim3 dimGrid( (SIZE + dimBlock.x - 1) / dimBlock.x, 1, 1 );
	CUDA_PRINT_CONFIG( SIZE );
	ELAPSED_TIME_BEGIN(0);
	kernelVecAdd <<< dimGrid, dimBlock>>>( dev_vecC, dev_vecA, dev_vecB, SIZE );
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
