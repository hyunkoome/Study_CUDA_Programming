#include "./common.cpp"
#include <limits.h>
#include <stdlib.h>
#include <algorithm>
#include <exception>
using namespace std;

unsigned num = 16 * 1024 * 1024; // maximum num of inputs
unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)

int compare(const void* lptr, const void* rptr) {
	int lhs = *(reinterpret_cast<const int*>(lptr));
	int rhs = *(reinterpret_cast<const int*>(rptr));
	return lhs - rhs;
}

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
	unsigned* ptr = static_cast<unsigned*>( bsearch(&targetValue, vecData, num, sizeof(unsigned), compare) );
	ELAPSED_TIME_END(0);
	// check the result
	if (ptr == nullptr) {
		printf("NOT FOUND: target value '%u' not found\n", targetValue);
	} else {
		unsigned index = ptr - vecData;
		printf("FOUND: vecData[%d] = %u\n", index, vecData[index]);
	}
	printVec( "vecData", vecData, num );
	// cleaning
	delete[] vecData;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
