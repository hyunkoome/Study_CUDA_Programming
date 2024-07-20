#include "./common.cpp"

const unsigned SIZE = 256 * 1024 * 1024; // big-size elements, 256M 256 백만 개

int main(void) {
	// host-side data
	float* vecA = nullptr;
	float* vecB = nullptr;
	float* vecC = nullptr;
	try {
		vecA = new float[SIZE];
		vecB = new float[SIZE];
		vecC = new float[SIZE];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(1);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecA, SIZE );
	setNormalizedRandomData( vecB, SIZE );
	// kernel execution
	ELAPSED_TIME_BEGIN(0);
	for (register unsigned i = 0; i < SIZE; ++i) {
		vecC[i] = vecA[i] + vecB[i];
	}
	ELAPSED_TIME_END(0);
	// check the result
	float sumA = getSum( vecA, SIZE );
	float sumB = getSum( vecB, SIZE );
	float sumC = getSum( vecC, SIZE );
	float diff = fabsf( sumC - (sumA + sumB) );
	printf("SIZE = %d\n", SIZE);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / SIZE =  %f\n", diff / SIZE);
	printVec( "vecA", vecA, SIZE );
	printVec( "vecB", vecB, SIZE );
	printVec( "vecC", vecC, SIZE );
	// cleaning
	delete[] vecA;
	delete[] vecB;
	delete[] vecC;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
