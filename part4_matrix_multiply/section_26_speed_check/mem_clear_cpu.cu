#include "./common.cpp"

// input parameters
unsigned num = 32 * 1024 * 1024; // num of data, 32M 개

int main( const int argc, const char* argv[] ) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1024 );
		break;
	default:
		printf("usage: %s [num]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("num=%u\n", num);
	// host-side data
	float* alpha = new float[num];
	float* bravo = new float[num];
	setNormalizedRandomData( alpha, num );
	setNormalizedRandomData( bravo, num );
	// show the original contents
	printVec( "alpha = ", alpha, num );
	printVec( "bravo = ", bravo, num );
	// clear with for-loop
	ELAPSED_TIME_BEGIN(0);
	for (register unsigned i = 0; i < num; ++i) {
		alpha[i] = 0.0f;
	}
	ELAPSED_TIME_END(0);
	// clear with memset
	ELAPSED_TIME_BEGIN(1);
	memset( bravo, 0, num * sizeof(float) );
	ELAPSED_TIME_END(1);
	// check them
	printVec( "alpha = ", alpha, num );
	printVec( "bravo = ", bravo, num );
	// cleaning
	delete[] alpha;
	delete[] bravo;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
