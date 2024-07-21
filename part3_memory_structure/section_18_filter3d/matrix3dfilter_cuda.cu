#include "./common.cpp"

// input parameters
dim3 dimImage( 300, 300, 256 ); // x, y, z order - width (ncolumn), height (nrow), depth

// CUDA kernel function
__global__ void kernel_filter( float* c, const float* a, const float* b,
                               unsigned ndim_z, unsigned ndim_y, unsigned ndim_x ) {
	unsigned idx_z = blockIdx.z * blockDim.z + threadIdx.z; // CUDA-provided index
	unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y; // CUDA-provided index
	unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (idx_x < ndim_x && idx_y < ndim_y && idx_z < ndim_z) {
		unsigned i = (idx_z * ndim_y + idx_y) * ndim_x + idx_x; // converted to 1D index
		c[i] = a[i] * b[i];
	}
}

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
		exit(EXIT_FAILURE);
	}
	// set random data
	srand( 0 );
	setNormalizedRandomData( matA, dimImage.z * dimImage.y * dimImage.x );
	setNormalizedRandomData( matB, dimImage.z * dimImage.y * dimImage.x );
	// device-side data
	float* dev_matA = nullptr;
	float* dev_matB = nullptr;
	float* dev_matC = nullptr;
	// allocate device memory
	ELAPSED_TIME_BEGIN(1);
	cudaMalloc( (void**)&dev_matA, dimImage.z * dimImage.y * dimImage.x * sizeof(float) );
	cudaMalloc( (void**)&dev_matB, dimImage.z * dimImage.y * dimImage.x * sizeof(float) );
	cudaMalloc( (void**)&dev_matC, dimImage.z * dimImage.y * dimImage.x * sizeof(float) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	cudaMemcpy( dev_matA, matA, dimImage.z * dimImage.y * dimImage.x * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_matB, matB, dimImage.z * dimImage.y * dimImage.x * sizeof(float), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(8, 8, 8);
	dim3 dimGrid(div_up(dimImage.x, dimBlock.x), div_up(dimImage.y, dimBlock.y), div_up(dimImage.z, dimBlock.z));
	CUDA_PRINT_CONFIG_3D( dimImage.x, dimImage.y, dimImage.z );
	ELAPSED_TIME_BEGIN(0);
	kernel_filter <<< dimGrid, dimBlock>>>( dev_matC, dev_matA, dev_matB, dimImage.z, dimImage.y, dimImage.x);
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( matC, dev_matC, dimImage.z * dimImage.y * dimImage.x * sizeof(float), cudaMemcpyDeviceToHost );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree( dev_matA );
	cudaFree( dev_matB );
	cudaFree( dev_matC );
	CUDA_CHECK_ERROR();
	// check the result
	float sumA = getSum( matA, dimImage.z * dimImage.y * dimImage.x);
	float sumB = getSum( matB, dimImage.z * dimImage.y * dimImage.x);
	float sumC = getSum( matC, dimImage.z * dimImage.y * dimImage.x);
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
