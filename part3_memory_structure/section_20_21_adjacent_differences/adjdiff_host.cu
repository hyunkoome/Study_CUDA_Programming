#include "./common.cpp"

// input parameters
unsigned num = 16 * 1024 * 1024; // num data

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1 );
		break;
	default:
		printf("usage: %s [num]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// host-side data
	float* vecA = nullptr;
	float* vecB = nullptr;
	try {
		vecA = new float[num];
		vecB = new float[num];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecA, num );
	// kernel execution
	ELAPSED_TIME_BEGIN(0);
	for (register unsigned i = 0; i < num; ++i) {
		if (i == 0) {
			vecB[i] = vecA[i] - 0.0f; // special case for i = 0
		} else {
			vecB[i] = vecA[i] - vecA[i - 1]; // normal case
		}
	}
	ELAPSED_TIME_END(0);
	// check the result
	float sumA = getSum( vecA, num );
	float sumB = getSum( vecB, num );
	printf("problem size = %d\n", num);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printVec( "vecA", vecA, num );
	printVec( "vecB", vecB, num );
	// cleaning
	delete[] vecA;
	delete[] vecB;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
