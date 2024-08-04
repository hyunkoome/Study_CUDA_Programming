#include "./common.cpp"
#include <algorithm>

// input parameters
const unsigned BLOCK_SIZE = 1024;
const unsigned TOTAL_NUM = 512 * (2 * BLOCK_SIZE); // max total num data
const unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)
enum {
	DECREASING = 0,
	INCREASING = 1,
};
unsigned direction = INCREASING;

template <int dir>
__device__ inline unsigned bsearchInclusive( uint32_t val, const uint32_t* pData, int bound, int stride ) {
	if (bound == 0) {
		return 0;
	}
	int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = min(pos + stride, bound);
		if ((dir && (pData[newPos - 1] <= val)) || (!dir && (pData[newPos - 1] >= val))) {
			pos = newPos;
		}
	}
	return pos;
}

template <int dir>
__device__ inline unsigned bsearchExclusive( uint32_t val, const uint32_t* pData, int bound, int stride ) {
	if (bound == 0) {
		return 0;
	}
	int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = umin(pos + stride, bound);
		if ((dir && (pData[newPos - 1] < val)) || (!dir && (pData[newPos - 1] > val))) {
			pos = newPos;
		}
	}
	return pos;
}

// CUDA kernel function
template <int dir>
__global__ void kernelMergeSortBlock( uint32_t* data ) {
	__shared__ uint32_t s_data[2 * BLOCK_SIZE]; // maximum shared memory size
	register unsigned tx = threadIdx.x;
	// copy from global memory to shared mem, (2 * dimBlock.x) elements
	register unsigned block_offset = 2 * blockIdx.x * blockDim.x;
	s_data[tx] = data[block_offset + tx];
	s_data[tx + blockDim.x] = data[block_offset + tx + blockDim.x];
	// merge sort main loop
	for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
		__syncthreads();
		int pos = tx & (stride - 1);
		uint32_t* base = s_data + 2 * (tx - pos);
		uint32_t keyA = base[pos + 0];
		uint32_t keyB = base[pos + stride];
		int posA = bsearchExclusive<dir>(keyA, base + stride, stride, stride) + pos;
		int posB = bsearchInclusive<dir>(keyB, base + 0, stride, stride) + pos;
		__syncthreads();
		base[posA] = keyA;
		base[posB] = keyB;
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
	dim3 dimGrid2(div_up(TOTAL_NUM, 2 * dimBlock.x) - 1, 1, 1);
	CUDA_PRINT_CONFIG( TOTAL_NUM );
	ELAPSED_TIME_BEGIN(0);
	if (direction == INCREASING) {
		for (unsigned i = 0; i < dimGrid.x; ++i) {
			kernelMergeSortBlock<INCREASING> <<< dimGrid, dimBlock>>>( dev_data );
			kernelMergeSortBlock<INCREASING> <<< dimGrid2, dimBlock>>>( dev_data + dimBlock.x );
		}
	} else {
		for (unsigned i = 0; i < dimGrid.x; ++i) {
			kernelMergeSortBlock<DECREASING> <<< dimGrid, dimBlock>>>( dev_data );
			kernelMergeSortBlock<DECREASING> <<< dimGrid2, dimBlock>>>( dev_data + dimBlock.x );
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
