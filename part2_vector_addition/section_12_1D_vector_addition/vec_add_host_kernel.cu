/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
#include "./common.cpp"

// cpu kernel 함수를 사용하는 방법

const unsigned SIZE = 1024 * 1024; // 1M elements

// kernel function: body of the FOR loop
void kernelVecAdd( unsigned i, float* c, const float* a, const float* b ) {
	c[i] = a[i] + b[i];
}

int main(void) {
	// host-side data
	float* vecA = new float[SIZE];
	float* vecB = new float[SIZE];
	float* vecC = new float[SIZE];
	// set random data
	srand( 0 );
	setNormalizedRandomData( vecA, SIZE );
	setNormalizedRandomData( vecB, SIZE );
	// kernel execution
	ELAPSED_TIME_BEGIN(0);

    // 나중에 cuda 로 바꿀것을 고려해서, 함수 body 만 kernel 함수로 뜯어냄
	for (register unsigned i = 0; i < SIZE; ++i) {
		kernelVecAdd( i, vecC, vecA, vecB );
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


