#include "./common.cpp"
#include <algorithm>

// input parameters
const unsigned BLOCK_SIZE = 1024;
const unsigned TOTAL_NUM = 16 * (2 * BLOCK_SIZE); // max total num data
unsigned bound = 1000 * 1000; // number will be ranged in [0..bound)
enum {
	DECREASING = 0,
	INCREASING = 1,
};
unsigned direction = INCREASING;

// CPU version: increasing order only
void merge( uint32_t* pData, int num ) {
	static uint32_t buf[2 * BLOCK_SIZE];
	int half = num / 2;
	uint32_t* lhs = pData;
	uint32_t* rhs = pData + half;
	int i = 0;
	int j = 0;
	int k = 0;
	while (i < half && j < half) {
		if (lhs[i] <= rhs[j]) {
			buf[k++] = lhs[i++];
		} else {
			buf[k++] = rhs[j++];
		}
	}
	while (i < half) {
		buf[k++] = lhs[i++];
	}
	while (j < half) {
		buf[k++] = rhs[j++];
	}
	memcpy( pData, buf, num * sizeof(uint32_t) );
}

// CPU version: increasing order only
void mergeSort( uint32_t* pData, int num ) {
	if (num > 1) {
		int half = num / 2;
		mergeSort( pData, half );
		mergeSort( pData + half, half );
		merge( pData, num );
	}
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
	// host-side data
	uint32_t* src = nullptr; // original data
	uint32_t* dst = nullptr; // processed
	try {
		src = new unsigned[TOTAL_NUM];
		dst = new unsigned[TOTAL_NUM];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setRandomData( src, TOTAL_NUM, bound );
	// configuration for each block
	unsigned unit_size = (2 * BLOCK_SIZE);
	unsigned num_units = TOTAL_NUM / unit_size;
	printf("UNIT SIZE = %d\n", unit_size);
	printf("NUM UNITS = %d\n", num_units);
	// CPU version
	memcpy( dst, src, TOTAL_NUM * sizeof(uint32_t) );
	ELAPSED_TIME_BEGIN(0);
	for (unsigned i = 0; i < num_units; ++i) {
		mergeSort( dst + i * (2 * BLOCK_SIZE), 2 * BLOCK_SIZE );
	}
	ELAPSED_TIME_END(0);
	// another processing with CPU
	for (unsigned i = 0; i < num_units; ++i) {
		std::sort( src + i * (2 * BLOCK_SIZE), src + (i + 1) * (2 * BLOCK_SIZE) );
	}
	// show the result
	uint32_t err = getTotalDiff( src, dst, TOTAL_NUM );
	printf("total diff = %d\n", err);
	printf("%d sorted units:\n", num_units);
	for (unsigned i = 0; i < num_units; ++i) {
		printVec( "dst", dst + i * (2 * BLOCK_SIZE), (2 * BLOCK_SIZE) );
	}
	// cleaning
	delete[] src;
	delete[] dst;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
