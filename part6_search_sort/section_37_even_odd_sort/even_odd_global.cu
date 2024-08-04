#include "./common.cpp"
#include <algorithm>

// input parameters
const unsigned BLOCK_SIZE = 1024;
const unsigned TOTAL_NUM = 16 * (2 * BLOCK_SIZE); // max total num data
const unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)
enum {
	DECREASING = 0,
	INCREASING = 1,
};
unsigned direction = INCREASING;

template <int dir, typename TYPE>
__device__ __host__ inline void comparator( TYPE& lhs, TYPE& rhs ) {
	// if dir == 0: decreasing case, we want to make (lhs >= rhs)
	// if dir == 1: increasing case, we want to make (lhs < rhs)
	if (dir == (lhs > rhs)) { // simple swap
		register TYPE t = lhs;
		lhs = rhs;
		rhs = t;
	}
}

template <int dir>
__global__ void kernelEvenOddGlobal_even( uint32_t* dst, unsigned num ) {
	register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
	if (2 * gx + 1 < num) {
		comparator<dir>( dst[2 * gx], dst[2 * gx + 1] );
	}
}

template <int dir>
__global__ void kernelEvenOddGlobal_odd( uint32_t* dst, unsigned num ) {
	register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gx != 0 && 2 * gx < num) {
		comparator<dir>( dst[2 * gx - 1], dst[2 * gx] );
	}
}

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		direction = procArg( argv[0], argv[1], 0, 1 );
		break;
	default:
		printf("usage: %s [direction] with 0=decreasing, 1=increasing\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("BLOCK_SIZE = %d, TOTAL_NUM = %d\n", BLOCK_SIZE, TOTAL_NUM);
	printf("bound = %d, dir = %s\n", bound, (direction == 0) ? "DECREASING" : "INCREASING" );
	// device check
	int count;
	cudaGetDeviceCount( &count );
	CUDA_CHECK_ERROR();
	if (count <= 0) {
		printf("%s: ERROR: no CUDA device\n", argv[0]);
		exit(0);
	}
	// host-side data
	uint32_t* src = nullptr; // original data, also processed by CPU
	uint32_t* dst = nullptr; // processed by CUDA
	try {
		src = new uint32_t[TOTAL_NUM];
		dst = new uint32_t[TOTAL_NUM];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setRandomData( src, TOTAL_NUM, bound );
	// device-side data
	uint32_t* dev_data = nullptr;
	cudaMalloc( (void**)&dev_data, TOTAL_NUM * sizeof(uint32_t) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy( dev_data, src, TOTAL_NUM * sizeof(uint32_t), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA 
	dim3 dimBlock(BLOCK_SIZE, 1, 1); // only one block
	dim3 dimGrid(div_up(TOTAL_NUM, 2 * dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( TOTAL_NUM );
	ELAPSED_TIME_BEGIN(0);
	if (direction == INCREASING) {
		for (register unsigned i = 0; i < TOTAL_NUM / 2; ++i) {
			kernelEvenOddGlobal_even<INCREASING> <<< dimGrid, dimBlock>>>( dev_data, TOTAL_NUM );
			kernelEvenOddGlobal_odd<INCREASING> <<< dimGrid, dimBlock>>>( dev_data, TOTAL_NUM );
		}
	} else {
		for (register unsigned i = 0; i < TOTAL_NUM / 2; ++i) {
			kernelEvenOddGlobal_even<DECREASING> <<< dimGrid, dimBlock>>>( dev_data, TOTAL_NUM );
			kernelEvenOddGlobal_odd<DECREASING> <<< dimGrid, dimBlock>>>( dev_data, TOTAL_NUM );
		}
	}
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( dst, dev_data, TOTAL_NUM * sizeof(uint32_t), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	// CPU version
	ELAPSED_TIME_BEGIN(1);
	if (direction == INCREASING) {
		std::sort( src, src + TOTAL_NUM );
	} else {
		std::sort( src, src + TOTAL_NUM, std::greater<uint32_t>() );
	}
	ELAPSED_TIME_END(1);
	// check the result
	uint32_t err = getTotalDiff( src, dst, TOTAL_NUM );
	printf("total diff = %d\n", err);
	printVec( "src", src, TOTAL_NUM );
	printVec( "dst", dst, TOTAL_NUM );
	// free device memory
	cudaFree( dev_data );
	CUDA_CHECK_ERROR();
	// cleaning
	delete[] src;
	delete[] dst;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
