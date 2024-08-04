#include "./common.cpp"
#include <algorithm>

// input parameters
const unsigned BLOCK_SIZE = 1024;
const unsigned TOTAL_NUM = 512 * (2 * 1024); // max total num data
const unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)
enum {
	DECREASING = 0,
	INCREASING = 1,
};
unsigned direction = INCREASING;

const unsigned SAMPLE_STRIDE = BLOCK_SIZE / 2;

template <typename TYPE>
__device__ __host__ inline void comparator( TYPE& lhs, TYPE& rhs, int dir ) {
	// if dir == 0: decreasing case, we want to make (lhs >= rhs)
	// if dir == 1: increasing case, we want to make (lhs < rhs)
	if (dir == (lhs > rhs)) { // simple swap
		register TYPE t = lhs;
		lhs = rhs;
		rhs = t;
	}
}

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
__device__ inline unsigned bsearchExclusive( unsigned val, const unsigned* pData, int bound, int stride ) {
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
__global__ void kernelMergeSortBlock( const uint32_t* src, uint32_t* dst ) {
	__shared__ unsigned s_data[2 * BLOCK_SIZE]; // maximum shared memory size
	register unsigned tx = threadIdx.x;
	// copy from global memory to shared mem, (2 * blockDim.x) elements
	register unsigned block_offset = 2 * blockIdx.x * blockDim.x;
	s_data[tx] = src[block_offset + tx];
	s_data[tx + blockDim.x] = src[block_offset + tx + blockDim.x];
	// merge sort main loop
	for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
		__syncthreads();
		int pos = tx & (stride - 1);
		uint32_t* base = s_data + 2 * (tx - pos);
		uint32_t keyA = base[pos + 0];
		uint32_t keyB = base[pos + stride];
		unsigned posA = bsearchExclusive<dir>(keyA, base + stride, stride, stride) + pos;
		unsigned posB = bsearchInclusive<dir>(keyB, base + 0, stride, stride) + pos;
		__syncthreads();
		base[posA] = keyA;
		base[posB] = keyB;
	}
	// copy from shared mem to global mem
	__syncthreads();
	dst[block_offset + tx] = s_data[tx];
	dst[block_offset + tx + blockDim.x] = s_data[tx + blockDim.x];
}


template <int dir>
__global__ void kernelGetRank( const uint32_t* pData, unsigned* rankA, unsigned* rankB, int segment_size ) {
	register unsigned bx = blockIdx.x; // target block number
	register unsigned tx = threadIdx.x; // sample number in a block
	register unsigned gx = tx + bx * blockDim.x;
	if (bx % 2 == 0) { // segment A
		rankA[gx] = tx * SAMPLE_STRIDE;
		const uint32_t* pSegA = pData + bx * segment_size; // start address of segment A
		const uint32_t* pSegB = pData + (bx + 1) * segment_size; // start address of segment B
		unsigned sampleValue = pSegA[tx * SAMPLE_STRIDE];
		rankB[gx] = bsearchExclusive<dir>( sampleValue, pSegB, segment_size, segment_size );
	} else { // segment B
		rankB[gx] = tx * SAMPLE_STRIDE;
		const uint32_t* pSegA = pData + (bx - 1) * segment_size; // start address of segment A
		const uint32_t* pSegB = pData + bx * segment_size; // start address of segment B
		unsigned sampleValue = pSegB[tx * SAMPLE_STRIDE];
		rankA[gx] = bsearchExclusive<dir>( sampleValue, pSegA, segment_size, segment_size );
	}
}

template <int dir>
__device__ inline void mergeIntervals( uint32_t* dst, const uint32_t* srcA, const uint32_t* srcB, 
			int lenA, int strideA, int lenB, int strideB ) {
	// assumes 'dst', 'srcA', 'srcB' have no overalps at all.
	register unsigned tx = threadIdx.x;
	register unsigned keyA, keyB, dstPosA, dstPosB;
	if (tx < lenA) {
		keyA = srcA[tx];
		dstPosA = bsearchExclusive<dir>(keyA, srcB, lenB, strideB) + tx;
	}
	if (tx < lenB) {
		keyB = srcB[tx];
		dstPosB = bsearchInclusive<dir>(keyB, srcA, lenA, strideA) + tx;
	}
	__syncthreads();
	if (tx < lenA) {
		dst[dstPosA] = keyA;
	}
	if (tx < lenB) {
		dst[dstPosB] = keyB;
	}
}

template <int dir>
__global__ void kernelMergeSegments( const uint32_t* src, uint32_t* dst,
                                     const unsigned* limitA, const unsigned* limitB, int segment_size ) {
	__shared__ unsigned s_src[2 * SAMPLE_STRIDE];
	__shared__ unsigned s_dst[2 * SAMPLE_STRIDE];
	register unsigned bx = blockIdx.x;
	register unsigned tx = threadIdx.x;
	register unsigned gx = tx + bx * blockDim.x;
	// get the segment base address
	register unsigned segBase = (gx / (2 * segment_size)) * (2 * segment_size);
	src += segBase;
	dst += segBase;
	__shared__ unsigned startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;
	if (threadIdx.x == 0) {
		unsigned num_samples = (2 * segment_size / SAMPLE_STRIDE);
		startSrcA = limitA[bx];
		startSrcB = limitB[bx];
		unsigned endSrcA = (bx % num_samples == num_samples - 1) ? segment_size : limitA[bx + 1];
		unsigned endSrcB = (bx % num_samples == num_samples - 1) ? segment_size : limitB[bx + 1];
		lenSrcA = endSrcA - startSrcA;
		lenSrcB = endSrcB - startSrcB;
		startDstA = startSrcA + startSrcB;
		startDstB = startDstA + lenSrcA;
#if 0
		if (segment_size == 2048 && bx < 16) {
			printf("bx = %d: segBase = %d startSrcA,B=%d,%d lenA,B=%d,%d startDstA,B=%d,%d\n",
			       bx, segBase, startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB);
		}
#endif
	}
	// get the data
	__syncthreads();
	if (tx < lenSrcA) {
		s_src[tx] = src[startSrcA + tx];
	}
	if (tx < lenSrcB) {
		s_src[tx + SAMPLE_STRIDE] = src[segment_size + startSrcB + tx];
	}
	// merge two intervals into one
	__syncthreads();
	mergeIntervals<dir>( s_dst, s_src, s_src + SAMPLE_STRIDE, lenSrcA, SAMPLE_STRIDE, lenSrcB, SAMPLE_STRIDE );
	// store the data
	__syncthreads();
	if (tx < lenSrcA) {
		dst[startDstA + tx] = s_dst[tx];
	}
	if (tx < lenSrcB) {
		dst[startDstB + tx] = s_dst[lenSrcA + tx];
	}
}


void deviceMergeSort( unsigned TOTAL_NUM, uint32_t* dev_vecA, uint32_t* dev_vecB, int dir ) {
	unsigned* rankA = nullptr; // sorted by CUDA
	unsigned* rankB = nullptr; // sorted by CUDA
	unsigned* limitA = nullptr; // sorted by CUDA
	unsigned* limitB = nullptr; // sorted by CUDA
	unsigned* vecC = nullptr; // sorted by CUDA
	rankA = new unsigned[TOTAL_NUM / SAMPLE_STRIDE];
	rankB = new unsigned[TOTAL_NUM / SAMPLE_STRIDE];
	limitA = new unsigned[TOTAL_NUM / SAMPLE_STRIDE];
	limitB = new unsigned[TOTAL_NUM / SAMPLE_STRIDE];
	vecC = new uint32_t[TOTAL_NUM];
	unsigned* dev_rankA = nullptr;
	unsigned* dev_rankB = nullptr;
	unsigned* dev_limitA = nullptr;
	unsigned* dev_limitB = nullptr;
	cudaMalloc( (void**)&dev_rankA, (TOTAL_NUM / SAMPLE_STRIDE) * sizeof(unsigned) );
	cudaMalloc( (void**)&dev_rankB, (TOTAL_NUM / SAMPLE_STRIDE) * sizeof(unsigned) );
	cudaMalloc( (void**)&dev_limitA, (TOTAL_NUM / SAMPLE_STRIDE) * sizeof(unsigned) );
	cudaMalloc( (void**)&dev_limitB, (TOTAL_NUM / SAMPLE_STRIDE) * sizeof(unsigned) );
	unsigned* dev_vecC = nullptr;
	cudaMalloc( (void**)&dev_vecC, TOTAL_NUM * sizeof(uint32_t) );
	CUDA_CHECK_ERROR();
	// step 0. sort each block
	dim3 dimBlock(BLOCK_SIZE, 1, 1); // only one block
	dim3 dimGrid(div_up(TOTAL_NUM, 2 * dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( TOTAL_NUM );
	if (dir) {
		kernelMergeSortBlock<INCREASING> <<< dimGrid, dimBlock>>>( dev_vecA, dev_vecB );
	} else {
		kernelMergeSortBlock<DECREASING> <<< dimGrid, dimBlock>>>( dev_vecA, dev_vecB );
	}
	CUDA_CHECK_ERROR();
	// main loop: 'segment_size' is size of each sorted segment
	for (unsigned segment_size = 2 * BLOCK_SIZE; segment_size < TOTAL_NUM; segment_size <<= 1) {
		// step 1. generate rank
		dimBlock = make_uint3( segment_size / SAMPLE_STRIDE, 1, 1 );
		dimGrid = make_uint3( TOTAL_NUM / segment_size, 1, 1 );
#if 0
		printf("rank: dimGrid  = %d * %d * %d, dimBlock = %d * %d * %d\n",
		         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
#endif
		if (dir) {
			kernelGetRank<INCREASING> <<< dimGrid, dimBlock>>>( dev_vecB, dev_rankA, dev_rankB, segment_size );
		} else {
			kernelGetRank<DECREASING> <<< dimGrid, dimBlock>>>( dev_vecB, dev_rankA, dev_rankB, segment_size );
		}
		CUDA_CHECK_ERROR();
		// step 2. calculate limit: actually sort two ranks as a cluster
		dimBlock = make_uint3( segment_size / SAMPLE_STRIDE, 1, 1 ); // merge sort gets two segments
		dimGrid = make_uint3( TOTAL_NUM / (2 * segment_size), 1, 1 );
#if 0
		CUDA_PRINT_CONFIG( TOTAL_NUM );
		printf("limit: dimGrid  = %d * %d * %d, dimBlock = %d * %d * %d\n",
		         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
#endif
		if (dir) {
			kernelMergeSortBlock<INCREASING> <<< dimGrid, dimBlock>>>( dev_rankA, dev_limitA );
			kernelMergeSortBlock<INCREASING> <<< dimGrid, dimBlock>>>( dev_rankB, dev_limitB );
		} else {
			kernelMergeSortBlock<DECREASING> <<< dimGrid, dimBlock>>>( dev_rankA, dev_limitA );
			kernelMergeSortBlock<DECREASING> <<< dimGrid, dimBlock>>>( dev_rankB, dev_limitB );
		}
		// step 3. merge two segments
		unsigned* src = dev_vecB;
		unsigned* dst = dev_vecC;
		dimBlock = make_uint3( SAMPLE_STRIDE, 1, 1 ); // merge gets two segments
		dimGrid = make_uint3( TOTAL_NUM / (1 * SAMPLE_STRIDE), 1, 1 );
#if 0
		CUDA_PRINT_CONFIG( TOTAL_NUM );
		printf("merge: dimGrid  = %d * %d * %d, dimBlock = %d * %d * %d\n",
		         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
#endif
		if (dir) {
			kernelMergeSegments<INCREASING> <<< dimGrid, dimBlock>>>( src, dst, dev_limitA, dev_limitB, segment_size );
		} else {
			kernelMergeSegments<DECREASING> <<< dimGrid, dimBlock>>>( src, dst, dev_limitA, dev_limitB, segment_size );
		}
		cudaMemcpy( vecC, dev_vecC, TOTAL_NUM * sizeof(unsigned), cudaMemcpyDeviceToHost );
		cudaMemcpy( dev_vecB, dev_vecC, TOTAL_NUM * sizeof(unsigned), cudaMemcpyDeviceToDevice );
		// partially done
		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR();
	}
	// done
	cudaFree( dev_rankA );
	cudaFree( dev_rankB );
	cudaFree( dev_limitA );
	cudaFree( dev_limitB );
	cudaFree( dev_vecC );
	CUDA_CHECK_ERROR();
	delete[] rankA;
	delete[] rankB;
	delete[] limitA;
	delete[] limitB;
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
	}
	// host-side data
	unsigned* src = nullptr; // original data
	unsigned* dst = nullptr; // sorted by CUDA
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
	unsigned* dev_vecA = nullptr;
	unsigned* dev_vecB = nullptr;
	cudaMalloc( (void**)&dev_vecA, TOTAL_NUM * sizeof(unsigned) );
	cudaMalloc( (void**)&dev_vecB, TOTAL_NUM * sizeof(unsigned) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy( dev_vecA, src, TOTAL_NUM * sizeof(unsigned), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA action kernel
	ELAPSED_TIME_BEGIN(0);
	deviceMergeSort( TOTAL_NUM, dev_vecA, dev_vecB, direction );
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( dst, dev_vecB, TOTAL_NUM * sizeof(unsigned), cudaMemcpyDeviceToHost );
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
	cudaFree( dev_vecA );
	cudaFree( dev_vecB );
	CUDA_CHECK_ERROR();
	// cleaning
	delete[] src;
	delete[] dst;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
