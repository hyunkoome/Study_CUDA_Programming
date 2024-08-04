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

template <typename TYPE>
__device__ __host__ inline void comparator_dir( TYPE& lhs, TYPE& rhs, int dir ) {
	// if dir == 0: decreasing case, we want to make (lhs >= rhs)
	// if dir == 1: increasing case, we want to make (lhs < rhs)
	if (dir == (lhs > rhs)) { // simple swap
		register TYPE t = lhs;
		lhs = rhs;
		rhs = t;
	}
}

// CUDA kernel function
template <int dir>
__global__ void kernelBitonicBlock( uint32_t* data ) {
	__shared__ uint32_t s_data[2 * BLOCK_SIZE]; // maximum shared memory size
	register unsigned tx = threadIdx.x;
	// copy from global memory to shared mem, (2 * blockDim.x) elements
	register unsigned block_offset = 2 * blockIdx.x * blockDim.x;
	s_data[tx] = data[block_offset + tx];
	s_data[tx + blockDim.x] = data[block_offset + tx + blockDim.x];
	// bitonic sort: 1st phase. halfsize is half of the target sorting range
	for (unsigned halfsize = 1; halfsize < blockDim.x; halfsize <<= 1) {
		// comparison steps
		unsigned cur_dir = dir ^ ((tx & halfsize) != 0);
		for (unsigned stride = halfsize; stride > 0; stride >>= 1) {
			__syncthreads();
			unsigned pos = 2 * tx - (tx & (stride - 1));
			comparator_dir( s_data[pos], s_data[pos + stride], cur_dir );
		}
	}
	// bitonic merge step
	for (unsigned stride = blockDim.x; stride > 0; stride >>= 1) {
		__syncthreads();
		unsigned pos = 2 * tx - (tx & (stride - 1));
		comparator<dir>( s_data[pos], s_data[pos + stride] );
	}
	// copy from shared mem to global mem
	__syncthreads();
	data[block_offset + tx] = s_data[tx];
	data[block_offset + tx + blockDim.x] = s_data[tx + blockDim.x];
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
	// set random data to A and B
	srand( 0 );
	setRandomData( src, TOTAL_NUM, bound );
	// device-side data
	uint32_t* dev_data = nullptr;
	cudaMalloc( (void**)&dev_data, TOTAL_NUM * sizeof(uint32_t) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy( dev_data, src, TOTAL_NUM * sizeof(uint32_t), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel
	dim3 dimBlock(BLOCK_SIZE, 1, 1); // only one block
	dim3 dimGrid(div_up(TOTAL_NUM, 2 * dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( TOTAL_NUM );
	ELAPSED_TIME_BEGIN(0);
	if (direction == INCREASING) {
		kernelBitonicBlock<INCREASING> <<< dimGrid, dimBlock>>>( dev_data );
	} else {
		kernelBitonicBlock<DECREASING> <<< dimGrid, dimBlock>>>( dev_data );
	}
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( dst, dev_data, TOTAL_NUM * sizeof(uint32_t), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	// CPU version
	if (direction == INCREASING) {
		for (unsigned i = 0; i < dimGrid.x; ++i) {
			std::sort( src + i * (2 * BLOCK_SIZE), src + (i + 1) * (2 * BLOCK_SIZE) );
		}
	} else {
		for (unsigned i = 0; i < dimGrid.x; ++i) {
			std::sort( src + i * (2 * BLOCK_SIZE), src + (i + 1) * (2 * BLOCK_SIZE), std::greater<uint32_t>() );
		}
	}
	// check the result
	uint32_t err = getTotalDiff( src, dst, TOTAL_NUM );
	printf("total diff = %d\n", err);
	printf("%d sorted blocks:\n", dimGrid.x);
	for (unsigned i = 0; i < dimGrid.x; ++i) {
		printVec( "dst", dst + i * (2 * BLOCK_SIZE), (2 * BLOCK_SIZE) );
	}
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
