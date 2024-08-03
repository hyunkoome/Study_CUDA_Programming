#include "./common.cpp"

// input parameters
const unsigned BLOCK_SIZE = 1024;
unsigned matsize = 16 * 1024; // num rows and also num cols
const float alpha = 0.012;
const float beta = 0.034;

// host-side data: Z = alpha A X + beta Y
float* vecZ = nullptr;
float* matA = nullptr;
float* vecX = nullptr;
float* vecY = nullptr;

// execute
void execute(void) {
	ELAPSED_TIME_BEGIN(0);
	// calculation
	for (register unsigned y = 0; y < matsize; ++y) {
		register float ans_ax = 0.0f;
		for (register unsigned k = 0; k < matsize; ++k) {
			unsigned indA = y * matsize + k; // convert to 1D index
			ans_ax += matA[indA] * vecX[k];
		}
		vecZ[y] = alpha * ans_ax + beta * vecY[y];
	}
	ELAPSED_TIME_END(0);
}

// check
void check(void) {
	// check the result
	float sumZ = getSum( vecZ, matsize );
	float sumA = getSum( matA, matsize * matsize );
	float sumX = getSum( vecX, matsize );
	float sumY = getSum( vecY, matsize );
	printf("sumZ = %f\n", sumZ);
	printf("sumA = %f\n", sumA);
	printf("sumX = %f\n", sumX);
	printf("sumY = %f\n", sumY);
	printVec( "vecZ", vecZ, matsize );
	printMat( "matA", matA, matsize, matsize );
	printVec( "vecX", vecX, matsize );
	printVec( "vecY", vecY, matsize );
}

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
	printf("PROBLEM SIZE: matsize = %d * %d\n", matsize, matsize);
	// host-side alloc
	try {
		vecZ = new float[matsize];
		matA = new float[matsize * matsize];
		vecX = new float[matsize];
		vecY = new float[matsize];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, matsize * matsize );
	setNormalizedRandomData( vecX, matsize );
	setNormalizedRandomData( vecY, matsize );
	// execute
	execute();
	check();
	// cleaning
	delete[] vecZ;
	delete[] matA;
	delete[] vecX;
	delete[] vecY;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
