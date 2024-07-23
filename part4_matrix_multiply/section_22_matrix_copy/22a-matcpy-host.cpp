#include "./common.cpp"

// input parameters
unsigned matsize = 4000; // num rows and also num cols

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		matsize = procArg( argv[0], argv[1], 4 );
		break;
	default:
		printf("usage: %s [matsize]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// host-side data
	float* matA = nullptr;
	float* matC = nullptr;
	try {
		matA = new float[matsize * matsize];
		matC = new float[matsize * matsize];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, matsize * matsize );
	// kernel processing
	ELAPSED_TIME_BEGIN(0);
	for (register int y = 0; y < matsize; ++y) {
		for (register int x = 0; x < matsize; ++x) {
			register unsigned i = y * matsize + x;
			matC[i] = matA[i];
		}
	}
	ELAPSED_TIME_END(0);
	// copy to host from device
	// free device memory
	// check the result
	float sumA = getSum( matA, matsize * matsize );
	float sumC = getSum( matC, matsize * matsize );
	float diff = fabsf( sumC - sumA );
	printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
	printf("sumA = %f\n", sumA);
	printf("sumC = %f\n", sumC);
	printf("diff(sumA, sumC) = %f\n", diff);
	printf("diff(sumA, sumC) / SIZE = %f\n", diff / (matsize * matsize));
	printMat( "matA", matA, matsize, matsize );
	printMat( "matC", matC, matsize, matsize );
	// cleaning
	delete[] matA;
	delete[] matC;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
