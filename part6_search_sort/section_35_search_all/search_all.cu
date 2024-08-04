#include "./common.cpp"
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
	printf("num = %u\n", num);
	printf("bound = %u\n", bound);
	// host-side data
	unsigned* vecData = nullptr;
	unsigned* vecIndex = nullptr;
	unsigned sizeIndex = num / bound * 4; // estimated max number of elements found
	try {
		vecData = new unsigned[num];
		vecIndex = new unsigned[sizeIndex];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set random data
	srand( 0 );
	setRandomData<unsigned>( vecData, num, bound );
	unsigned targetValue = vecData[num - 1];
	printf("target value = %u\n", targetValue);
	// kernel
	ELAPSED_TIME_BEGIN(0);
	unsigned found = 0;
	for (unsigned i = 0; i < num; ++i) {
		if (vecData[i] == targetValue) {
			vecIndex[found] = i;
			found++;
		}
	}
	ELAPSED_TIME_END(0);
	// check the result
	printf("%d locations are found\n", found);
	for (unsigned i = 0; i < found; ++i) {
		unsigned index = vecIndex[i];
		printf("vecData[%d]= %d\n", index, vecData[index]);
	}
	printVec( "vecIndex", vecIndex, found );
	printVec( "vecData", vecData, num );
	// cleaning
	delete[] vecData;
	delete[] vecIndex;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
