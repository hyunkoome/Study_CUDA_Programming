#include "./common.cpp"

// input parameters
dim3 dimImage( 300, 300, 256 ); // x, y, z order - width (ncolumn), height (nrow), depth

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		dimImage.x = dimImage.y = dimImage.z = procArg( argv[0], argv[1], 32, 1024 );
		break;
	case 4:
		dimImage.z = procArg( argv[0], argv[1], 32, 1024 );
		dimImage.y = procArg( argv[0], argv[2], 32, 1024 );
		dimImage.x = procArg( argv[0], argv[3], 32, 1024 );
		break;
	default:
		printf("usage: %s [dim.z [dim.y dim.x]]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// host-side data
	float* matA = nullptr;
	float* matB = nullptr;
	float* matC = nullptr;
	try {
		matA = new float[dimImage.z * dimImage.y * dimImage.x];
		matB = new float[dimImage.z * dimImage.y * dimImage.x];
		matC = new float[dimImage.z * dimImage.y * dimImage.x];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, dimImage.z * dimImage.y * dimImage.x );
	setNormalizedRandomData( matB, dimImage.z * dimImage.y * dimImage.x );
	// kernel execution
	ELAPSED_TIME_BEGIN(0);
	for (register unsigned z = 0; z < dimImage.z; ++z) {
		for (register unsigned y = 0; y < dimImage.y; ++y) {
			for (register unsigned x = 0; x < dimImage.x; ++x) {
				unsigned i = (z * dimImage.y + y) * dimImage.x + x; // convert to 1D index
				matC[i] = matA[i] * matB[i];
			}
		}
	}
	ELAPSED_TIME_END(0);
	// check the result
	float sumA = getSum( matA, dimImage.z * dimImage.y * dimImage.x );
	float sumB = getSum( matB, dimImage.z * dimImage.y * dimImage.x );
	float sumC = getSum( matC, dimImage.z * dimImage.y * dimImage.x );
	printf("matrix size = %d * %d * %d\n", dimImage.z, dimImage.y, dimImage.x);
	printf("sumC = %f\n", sumC);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	print3D( "C", matC, dimImage.z, dimImage.y, dimImage.x );
	print3D( "A", matA, dimImage.z, dimImage.y, dimImage.x );
	print3D( "B", matB, dimImage.z, dimImage.y, dimImage.x );
	// cleaning
	delete[] matA;
	delete[] matB;
	delete[] matC;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
