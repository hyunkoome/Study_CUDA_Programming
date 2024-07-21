#include "./common.cpp"

// input parameters
dim3 dimImage( 300, 300, 256 ); // x, y, z order - width (ncolumn), height (nrow), depth

// CUDA kernel function
__global__ void kernel_filter( void* matC, const void* matA, const void* matB,
                               size_t pitch, unsigned ndim_z, unsigned ndim_y, unsigned ndim_x ) {
	register unsigned idx_z = blockIdx.z * blockDim.z + threadIdx.z; // CUDA-provided index
	register unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y; // CUDA-provided index
	register unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (idx_x < ndim_x && idx_y < ndim_y && idx_z < ndim_z) {
		register unsigned offset_in_byte = (idx_z * ndim_y + idx_y) * pitch + idx_x * sizeof(float);
		*((float*)((char*)matC + offset_in_byte))
		    = *((const float*)((const char*)matA + offset_in_byte))
		      * *((const float*)((const char*)matB + offset_in_byte));
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
	struct cudaExtent extentInByte = make_cudaExtent( dimImage.x * sizeof(float), dimImage.y, dimImage.z );
	struct cudaPitchedPtr dev_pitchedA = { 0 };
	struct cudaPitchedPtr dev_pitchedB = { 0 };
	struct cudaPitchedPtr dev_pitchedC = { 0 };
	// allocate device memory
	ELAPSED_TIME_BEGIN(1);
	cudaMalloc3D( &dev_pitchedA, extentInByte );
	cudaMalloc3D( &dev_pitchedB, extentInByte );
	cudaMalloc3D( &dev_pitchedC, extentInByte );
	CUDA_CHECK_ERROR();
	printf("dev_pitchedA=%p pitch=%zd xsize=%zd ysize=%zd\n",
	       dev_pitchedA.ptr, dev_pitchedA.pitch, dev_pitchedA.xsize, dev_pitchedA.ysize);
	printf("dev_pitchedB=%p pitch=%zd xsize=%zd ysize=%zd\n",
	       dev_pitchedB.ptr, dev_pitchedB.pitch, dev_pitchedB.xsize, dev_pitchedB.ysize);
	printf("dev_pitchedC=%p pitch=%zd xsize=%zd ysize=%zd\n",
	       dev_pitchedC.ptr, dev_pitchedC.pitch, dev_pitchedC.xsize, dev_pitchedC.ysize);
	// copy to device from host
	struct cudaPitchedPtr pitchedA
	    = make_cudaPitchedPtr( matA, dimImage.x * sizeof(float), dimImage.x * sizeof(float), dimImage.y );
	struct cudaPitchedPtr pitchedB
	    = make_cudaPitchedPtr( matB, dimImage.x * sizeof(float), dimImage.x * sizeof(float), dimImage.y );
	struct cudaPitchedPtr pitchedC
	    = make_cudaPitchedPtr( matC, dimImage.x * sizeof(float), dimImage.x * sizeof(float), dimImage.y );
	struct cudaPos pos_origin = make_cudaPos( 0, 0, 0 );
	struct cudaMemcpy3DParms paramA = { 0 };
	struct cudaMemcpy3DParms paramB = { 0 };
	struct cudaMemcpy3DParms paramC = { 0 };
	printf("pitchedA=%p pitch=%zd xsize=%zd ysize=%zd\n",
	       pitchedA.ptr, pitchedA.pitch, pitchedA.xsize, pitchedA.ysize);
	printf("pitchedB=%p pitch=%zd xsize=%zd ysize=%zd\n",
	       pitchedB.ptr, pitchedB.pitch, pitchedB.xsize, pitchedB.ysize);
	printf("pitchedC=%p pitch=%zd xsize=%zd ysize=%zd\n",
	       pitchedC.ptr, pitchedC.pitch, pitchedC.xsize, pitchedC.ysize);
	paramA.srcPos = pos_origin;
	paramA.srcPtr = pitchedA;
	paramA.dstPos = pos_origin;
	paramA.dstPtr = dev_pitchedA;
	paramA.extent = extentInByte;
	paramA.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D( &paramA );
	paramB.srcPos = pos_origin;
	paramB.srcPtr = pitchedB;
	paramB.dstPos = pos_origin;
	paramB.dstPtr = dev_pitchedB;
	paramB.extent = extentInByte;
	paramB.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D( &paramB );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(8, 8, 8);
	dim3 dimGrid(div_up(dimImage.x, dimBlock.x), div_up(dimImage.y, dimBlock.y), div_up(dimImage.z, dimBlock.z));
	CUDA_PRINT_CONFIG_3D( dimImage.x, dimImage.y, dimImage.z );
	ELAPSED_TIME_BEGIN(0);
	kernel_filter <<< dimGrid, dimBlock>>>( dev_pitchedC.ptr, dev_pitchedA.ptr, dev_pitchedB.ptr,
	                                        dev_pitchedA.pitch, dimImage.z, dimImage.y, dimImage.x );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	paramC.srcPos = pos_origin;
	paramC.srcPtr = dev_pitchedC;
	paramC.dstPos = pos_origin;
	paramC.dstPtr = pitchedC;
	paramC.extent = extentInByte;
	paramC.kind = cudaMemcpyDeviceToHost;
	cudaMemcpy3D( &paramC );
	CUDA_CHECK_ERROR();
	ELAPSED_TIME_END(1);
	// free device memory
	cudaFree( dev_pitchedA.ptr );
	cudaFree( dev_pitchedB.ptr );
	cudaFree( dev_pitchedC.ptr );
	CUDA_CHECK_ERROR();
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
