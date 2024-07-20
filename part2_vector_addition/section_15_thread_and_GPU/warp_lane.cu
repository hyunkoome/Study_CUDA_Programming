#include "./common.cpp"

__device__ unsigned lane_id(void) {
	unsigned ret;
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ unsigned warp_id(void) {
	// this is not equal to threadIdx.x / 32
	unsigned ret;
	asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
	return ret;
}

__global__ void kernel_warp_lane( void ) {
	unsigned warpid = warp_id();
	unsigned laneid = lane_id();
	if (warpid == 0) {
		printf("lane=%2u threadIdx.x=%2d threadIdx.y=%2d blockIdx.x=%2d blockIdx.y=%2d\n",
		       laneid, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
	}
}

int main( void ) {
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(2, 2, 1);
	kernel_warp_lane <<< dimGrid, dimBlock>>>();
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
	// done
	return 0;
}

/* (c) 2021-2022. biztripcru@gmail.com. All rights reserved. */
