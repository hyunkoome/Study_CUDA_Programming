#include "./common.cpp"

// input parameters
const unsigned MAX_NUM = 16 * 1024 * 1024;
unsigned NUM = MAX_NUM; // num data

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		NUM = procArg( argv[0], argv[1], 1 );
		break;
	default:
		printf("usage: %s [num] [blocksize]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("NUM = %d\n", NUM);
	// host-side data
	float* vecData = nullptr;
	try {
		vecData = new float[NUM];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecData, NUM );
	// kernel execution
	ELAPSED_TIME_BEGIN(0);
	float sumData = getSum( vecData, NUM );
	ELAPSED_TIME_END(0);
	// check the result
	printVec( "data", vecData, NUM );
	printf("sum(data) = %f from CPU processing\n", sumData);
	// cleaning
	delete[] vecData;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
