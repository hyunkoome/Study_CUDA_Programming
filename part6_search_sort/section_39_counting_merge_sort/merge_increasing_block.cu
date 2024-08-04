#include "./common.cpp"
#include <algorithm>

// input parameters
const unsigned BLOCK_SIZE = 1024;
const unsigned TOTAL_NUM = 16 * (2* BLOCK_SIZE); // max total num data
unsigned bound = 1000 * 1000; // number will be ranged in [0..bound)
enum {
	DECREASING = 0,
	INCREASING = 1,
};
unsigned direction = INCREASING;

__device__ inline int bsearchInclusive( uint32_t val, const uint32_t* pData, int bound, int stride ) {
	if (bound == 0) {
		return 0;
	}
	int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = min(pos + stride, bound);
		if (pData[newPos - 1] <= val) {
			pos = newPos;
		}
	}
	return pos;
}

__device__ inline int bsearchExclusive( uint32_t val, const uint32_t* pData, int bound, int stride ) {
	if (bound == 0) {
		return 0;
	}
	int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = umin(pos + stride, bound);
		if (pData[newPos - 1] < val) {
			pos = newPos;
		}
	}
	return pos;
}

// CUDA kernel function
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
		int posA = bsearchExclusive(keyA, base + stride, stride, stride) + pos;
		int posB = bsearchInclusive(keyB, base + 0, stride, stride) + pos;
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
	default:
		printf("usage: %s\n", argv[0]);
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
	unsigned* src = nullptr; // original data, also processed by CPU
	unsigned* dst = nullptr; // processed by CUDA
	try {
		src = new unsigned[TOTAL_NUM];
		dst = new unsigned[TOTAL_NUM];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data to A and B
	srand( 0 );
	setRandomData( src, TOTAL_NUM, bound );
	// device-side data
	unsigned* dev_data = nullptr;
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
	kernelMergeSortBlock <<< dimGrid, dimBlock>>>( dev_data );
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
