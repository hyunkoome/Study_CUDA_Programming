#include "./common.cpp"
#include <limits.h>
#include <algorithm>
#include <exception>
using namespace std;

unsigned num = 16 * 1024 * 1024; // maximum num of inputs
unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1024 );
		break;
	case 3:
		num = procArg( argv[0], argv[1], 1024 );
		bound = procArg( argv[0], argv[2], 1024 );
		break;
	default:
		printf("usage: %s [num] [bound]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("num = %d, bound = %d\n", num, bound);
	// host-side data
	unsigned* vecData = nullptr;
	try {
		vecData = new unsigned[num];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set random data
	srand( 0 );
	setRandomData( vecData, num, bound);
	unsigned targetValue = vecData[num - 1];
	printf("target value = %u\n", targetValue);
	// we need to sort it, for the binary search
	std::sort(vecData, vecData + num);
	// kernel
	ELAPSED_TIME_BEGIN(0);
	bool flag = std::binary_search( vecData, vecData + num, targetValue );
	ELAPSED_TIME_END(0);
	// check the result
	if (flag == false) {
		printf("NOT FOUND: target value '%u' not found\n", targetValue);
	} else {
		printf("FOUND: vecData: %d found\n", targetValue);
	}
	printVec( "vecData", vecData, num );
	// kernel again
	ELAPSED_TIME_BEGIN(1);
	unsigned* lptr = std::lower_bound( vecData, vecData + num, targetValue );
	unsigned* uptr = std::upper_bound( vecData, vecData + num, targetValue );
	if (lptr == uptr) {
		printf("NOT FOUND: target value '%u' not found\n", targetValue);
	} else {
		int lower = lptr - vecData;
		int upper = uptr - vecData;
		printf("FOUND: %d elements found\n", upper - lower);
		printf("lower: vecData[%d] = %u\n", lower, vecData[lower]);
		printf("upper: vecData[%d] = %u\n", upper, vecData[upper]);
	}
	ELAPSED_TIME_END(1);
	// one more time
	ELAPSED_TIME_BEGIN(2);
	auto pair = std::equal_range( vecData, vecData + num, targetValue );
	lptr = pair.first;
	uptr = pair.second;
	if (lptr == uptr) {
		printf("NOT FOUND: target value '%u' not found\n", targetValue);
	} else {
		int lower = lptr - vecData;
		int upper = uptr - vecData;
		printf("FOUND: %d elements found\n", upper - lower);
		printf("lower: vecData[%d] = %u\n", lower, vecData[lower]);
		printf("upper: vecData[%d] = %u\n", upper, vecData[upper]);
	}
	ELAPSED_TIME_END(2);
	// cleaning
	delete[] vecData;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
