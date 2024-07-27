#include "./common.cpp"

// input parameters
unsigned matsize = 1024; // num rows and also num cols

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
	float* matB = nullptr;
	float* matC = nullptr;
	try {
		matA = new float[matsize * matsize];
		matB = new float[matsize * matsize];
		matC = new float[matsize * matsize];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, matsize * matsize );
	setNormalizedRandomData( matB, matsize * matsize );
	// kernel execution
	ELAPSED_TIME_BEGIN(0);
	memset( matC, 0, matsize * matsize * sizeof(float) );
	for (register unsigned k = 0; k < matsize; ++k) {
		for (register unsigned y = 0; y < matsize; ++y) {
			for (register unsigned x = 0; x < matsize; ++x) {
				unsigned indC = y * matsize + x; // convert to 1D index
				unsigned indA = y * matsize + k; // convert to 1D index
				unsigned indB = k * matsize + x; // convert to 1D index
				matC[indC] += matA[indA] * matB[indB];
			}
		}
	}
	ELAPSED_TIME_END(0);
	// check the result
	float sumA = getSum( matA, matsize * matsize );
	float sumB = getSum( matB, matsize * matsize );
	float sumC = getSum( matC, matsize * matsize );
	printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printMat( "matC", matC, matsize, matsize );
	printMat( "matA", matA, matsize, matsize );
	printMat( "matB", matB, matsize, matsize );
	// cleaning
	delete[] matA;
	delete[] matB;
	delete[] matC;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
