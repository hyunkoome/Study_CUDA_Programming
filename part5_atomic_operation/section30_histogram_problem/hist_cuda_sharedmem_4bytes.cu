#include "./common.cpp"
#include "./image.cpp"

const unsigned image_width = 640;
const unsigned image_height = 400;
const unsigned HIST_SIZE = 32; // histogram levels

// CUDA kernel function
__global__ void kernelHist(const unsigned* img, unsigned num, unsigned* hist) {
	__shared__ unsigned s_hist[HIST_SIZE];
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (threadIdx.x < HIST_SIZE) {
		s_hist[threadIdx.x] = 0;
	}
	__syncthreads();
	if (i < num) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 // CAUTION: atomicAdd_block(): CUDA arch 6.0 or later required.
#error CUDA arch 6.0 or later required. Use '-arch sm_60' or higher, to compile this program
#endif
		unsigned pixelVal = img[i];
		atomicAdd_block( &(s_hist[((pixelVal >> 0) & 0xFF) / 8]), 1 );
		atomicAdd_block( &(s_hist[((pixelVal >> 8) & 0xFF) / 8]), 1 );
		atomicAdd_block( &(s_hist[((pixelVal >> 16) & 0xFF) / 8]), 1 );
		atomicAdd_block( &(s_hist[((pixelVal >> 24) & 0xFF) / 8]), 1 );
	}
	__syncthreads();
	if (threadIdx.x < HIST_SIZE) {
		atomicAdd( &(hist[threadIdx.x]), s_hist[threadIdx.x] );
	}
}

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	default:
		printf("usage: %s\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	// host-side data
	unsigned* vecHist = nullptr;
	try {
		vecHist = new unsigned[HIST_SIZE];
	} catch (const exception& e) {
		printf("C++ EXCEPTION: %s\n", e.what());
		exit(EXIT_FAILURE); // ENOMEM: cannot allocate memory
	}
	// set data to be zero
	memset(vecHist, 0, HIST_SIZE * sizeof(unsigned));
	// device-side data
	unsigned char* dev_image = nullptr;
	unsigned* dev_vecHist = nullptr;
	cudaMalloc( (void**)&dev_image, sizeof(grayscale_data) );
	cudaMalloc( (void**)&dev_vecHist, HIST_SIZE * sizeof(unsigned) );
	CUDA_CHECK_ERROR();
	// copy to device from host
	ELAPSED_TIME_BEGIN(1);
	cudaMemcpy( dev_image, grayscale_data, sizeof(grayscale_data), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_vecHist, vecHist, HIST_SIZE * sizeof(unsigned), cudaMemcpyHostToDevice );
	CUDA_CHECK_ERROR();
	// CUDA kernel launch
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(div_up(image_width * image_height / 4, dimBlock.x), 1, 1);
	CUDA_PRINT_CONFIG( image_width * image_height / 4 );
	ELAPSED_TIME_BEGIN(0);
	kernelHist <<< dimGrid, dimBlock>>>( reinterpret_cast<unsigned*>(dev_image),
	                                     (image_width * image_height / 4), dev_vecHist );
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);
	CUDA_CHECK_ERROR();
	// copy to host from device
	cudaMemcpy( vecHist, dev_vecHist, HIST_SIZE * sizeof(unsigned), cudaMemcpyDeviceToHost );
	ELAPSED_TIME_END(1);
	CUDA_CHECK_ERROR();
	// free device memory
	cudaFree( dev_image );
	cudaFree( dev_vecHist );
	CUDA_CHECK_ERROR();
	// check the result
	printf("image pixels = %zu\n", sizeof(grayscale_data));
	printf("histogram levels = %u\n", HIST_SIZE);
	unsigned sum = 0;
	for (register unsigned i = 0; i < HIST_SIZE; ++i) {
		printf("hist[%2d] = %8u\n", i, vecHist[i]);
		sum += vecHist[i];
	}
	printf("sum = %u\n", sum);
	// cleaning
	delete[] vecHist;
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
