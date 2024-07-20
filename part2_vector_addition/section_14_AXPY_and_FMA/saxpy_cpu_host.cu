#include "./common.cpp"

// input parameters
unsigned vecSize = 256 * 1024 * 1024; // big-size elements
float saxpy_a = 1.234f;

int main( const int argc, const char* argv[] ) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		vecSize = procArg( argv[0], argv[1], 1 );
		break;
	case 3:
		vecSize = procArg( argv[0], argv[1], 1 );
		saxpy_a = procArg<float>( argv[0], argv[2] );
		break;
	default:
		printf("usage: %s [num] [a]\n", argv[0]);
		exit( EXIT_FAILURE );
		break;
	}
	// host-side data
	float* vecX = nullptr;
	float* vecY = nullptr;
	float* vecZ = nullptr;
	try {
		vecX = new float[vecSize];
		vecY = new float[vecSize];
		vecZ = new float[vecSize];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(1);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecX, vecSize );
	setNormalizedRandomData( vecY, vecSize );
	// kernel execution
	ELAPSED_TIME_BEGIN(0);
	for (register unsigned i = 0; i < vecSize; ++i) {
		vecZ[i] = saxpy_a * vecX[i] + vecY[i];
	}
	ELAPSED_TIME_END(0);
	// check the result
	float sumX = getSum( vecX, vecSize );
	float sumY = getSum( vecY, vecSize );
	float sumZ = getSum( vecZ, vecSize );
	float diff = fabsf( sumZ - (saxpy_a * sumX + sumY) );
	printf("SIZE = %d\n", vecSize);
	printf("a    = %f\n", saxpy_a);
	printf("sumX = %f\n", sumX);
	printf("sumY = %f\n", sumY);
	printf("sumZ = %f\n", sumZ);
	printf("diff(sumZ, a*sumX+sumY) =  %f\n", diff);
	printf("diff(sumZ, a*sumX+sumY)/SIZE =  %f\n", diff / vecSize);
	printVec( "vecX", vecX, vecSize );
	printVec( "vecY", vecY, vecSize );
	printVec( "vecZ", vecZ, vecSize );
	// cleaning
	delete[] vecX;
	delete[] vecY;
	delete[] vecZ;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
