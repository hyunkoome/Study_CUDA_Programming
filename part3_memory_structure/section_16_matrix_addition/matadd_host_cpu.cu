#include "./common.cpp"

// input parameters
unsigned nrow = 10000; // num rows
unsigned ncol = 10000; // num columns

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		nrow = ncol = procArg( argv[0], argv[1], 32, 16 * 1024 );
		break;
	case 3:
		nrow = procArg( argv[0], argv[1], 32, 16 * 1024 );
		ncol = procArg( argv[0], argv[2], 32, 16 * 1024 );
		break;
	default:
		printf("usage: %s [nrow] [ncol]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// host-side data
	float* matA = nullptr;
	float* matB = nullptr;
	float* matC = nullptr;
	try {
		matA = new float[nrow * ncol]; // nrow * ncol : 1억개
		matB = new float[nrow * ncol];
		matC = new float[nrow * ncol];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, nrow * ncol );
	setNormalizedRandomData( matB, nrow * ncol );
	// kernel execution
	ELAPSED_TIME_BEGIN(0);
	for (register unsigned r = 0; r < nrow; ++r) {
		for (register unsigned c = 0; c < ncol; ++c) {
			unsigned i = r * ncol + c; // convert to 1D index
			matC[i] = matA[i] + matB[i];
		}
	}
	ELAPSED_TIME_END(0);
	// check the result
	float sumA = getSum( matA, nrow * ncol );
	float sumB = getSum( matB, nrow * ncol );
	float sumC = getSum( matC, nrow * ncol );
	float diff = fabsf( sumC - (sumA + sumB) );
	printf("matrix size = nrow * ncol = %d * %d\n", nrow, ncol);
	printf("sumC = %f\n", sumC);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / (nrow * ncol) =  %f\n", diff / (nrow * ncol));
	printMat( "matC", matC, nrow, ncol );
	printMat( "matA", matA, nrow, ncol );
	printMat( "matB", matB, nrow, ncol );
	// cleaning
	delete[] matA;
	delete[] matB;
	delete[] matC;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
