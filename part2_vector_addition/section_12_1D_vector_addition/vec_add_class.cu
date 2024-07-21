#include "./common.cpp"

// CUDA kernel function
template<typename T>
__global__ void kernelVecAdd( T* c, const T* a, const T* b, unsigned n ) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

class VecAdd {
protected:
	const unsigned SIZE = 1024 * 1024; // 1M elements
	// host-side
	float* vecA;
	float* vecB;
	float* vecC;
	// device-side
	float* dev_vecA;
	float* dev_vecB;
	float* dev_vecC;
public:
	void prepare_host(void) {
		// host-side data
		vecA = new float[SIZE];
		vecB = new float[SIZE];
		vecC = new float[SIZE];
		// set random data
		srand( 0 );
		setNormalizedRandomData( vecA, SIZE );
		setNormalizedRandomData( vecB, SIZE );
	}
	void copy_to_device(void) {
		// allocate device memory
		cudaMalloc( (void**)&dev_vecA, SIZE * sizeof(float) );
		cudaMalloc( (void**)&dev_vecB, SIZE * sizeof(float) );
		cudaMalloc( (void**)&dev_vecC, SIZE * sizeof(float) );
		// copy to device from host
		cudaMemcpy( dev_vecA, vecA, SIZE * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( dev_vecB, vecB, SIZE * sizeof(float), cudaMemcpyHostToDevice );
		CUDA_CHECK_ERROR();
	}
	void execute_kernel(void) {
		// kernel launch
		kernelVecAdd<float> <<< SIZE / 1024, 1024>>>( dev_vecC, dev_vecA, dev_vecB, SIZE );
		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR();
	}
	void copy_to_host(void) {
		// copy to host from device
		cudaMemcpy( vecC, dev_vecC, SIZE * sizeof(float), cudaMemcpyDeviceToHost );
		CUDA_CHECK_ERROR();
	}
	void check(void) {
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
	}
	void clear(void) {
		// free device memory
		cudaFree( dev_vecA );
		cudaFree( dev_vecB );
		cudaFree( dev_vecC );
		CUDA_CHECK_ERROR();
		// cleaning
		delete[] vecA;
		delete[] vecB;
		delete[] vecC;
	}
};


int main(void) {
	VecAdd vecadd;
	vecadd.prepare_host();
	ELAPSED_TIME_BEGIN(1);
	vecadd.copy_to_device();
	ELAPSED_TIME_BEGIN(0);
	vecadd.execute_kernel();
	ELAPSED_TIME_END(0);
	vecadd.copy_to_host();
	ELAPSED_TIME_END(1);
	vecadd.check();
	vecadd.clear();
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
