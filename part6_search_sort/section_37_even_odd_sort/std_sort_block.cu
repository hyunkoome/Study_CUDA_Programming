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

int compLess( const void* lhs, const void* rhs ) {
	uint32_t lval = *(static_cast<const uint32_t*>(lhs));
	uint32_t rval = *(static_cast<const uint32_t*>(rhs));
	return (lval - rval);
}

int compGreater( const void* lhs, const void* rhs ) {
	uint32_t lval = *(static_cast<const uint32_t*>(lhs));
	uint32_t rval = *(static_cast<const uint32_t*>(rhs));
	return (rval - lval);
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
	// host-side data
	uint32_t* src = nullptr; // original data
	uint32_t* dst = nullptr; // processed by CPU
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
	// configuration for each block
	unsigned unit_size = (2 * BLOCK_SIZE);
	unsigned num_units = TOTAL_NUM / unit_size;
	printf("UNIT SIZE = %d\n", unit_size);
	printf("NUM UNITS = %d\n", num_units);
	// CPU processing
	memcpy( dst, src, TOTAL_NUM * sizeof(uint32_t) );
	ELAPSED_TIME_BEGIN(0);
	if (direction == INCREASING) {
		for (unsigned i = 0; i < num_units; ++i) {
			std::sort( dst + i * (2 * BLOCK_SIZE), dst + (i + 1) * (2 * BLOCK_SIZE) );
		}
	} else {
		for (unsigned i = 0; i < num_units; ++i) {
			std::sort( dst + i * (2 * BLOCK_SIZE), dst + (i + 1) * (2 * BLOCK_SIZE), std::greater<uint32_t>() );
		}
	}
	ELAPSED_TIME_END(0);
	// another processing with CPU
	if (direction == INCREASING) {
		for (unsigned i = 0; i < num_units; ++i) {
			qsort( src + i * (2 * BLOCK_SIZE), (2 * BLOCK_SIZE), sizeof(uint32_t), compLess );
		}
	} else {
		for (unsigned i = 0; i < num_units; ++i) {
			qsort( src + i * (2 * BLOCK_SIZE), (2 * BLOCK_SIZE), sizeof(uint32_t), compGreater );
		}
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
