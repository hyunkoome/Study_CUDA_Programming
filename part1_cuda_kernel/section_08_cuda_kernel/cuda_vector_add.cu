//
// Created by hyunkoo on 24. 7. 16.
//
#include <cuda.h> // bisic header file
#include <cuda_runtime_api.h> // compiler 사용시, runtime function을 사용가능, API = application programming interface (함수 정의)
#include <cuda_runtime.h> // CUDA built-ins, types 등 사용 가능
#include <stdio.h>
#include "./common.cpp"

// kernel program for the device (GPU): compiled by NVCC
__global__ void add_kernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x; // each thread know its own index, CUDA kernel에서 index 변수 자동 설정
    c[i] = a[i] + b[i];
}

// main program for the CPU: compiled by GCC
int main(void) {
    // host-side data
    const int SIZE = 5;
    const int a[SIZE] = {1, 2, 3, 4, 5};
    const int b[SIZE] = {10, 20, 30, 40, 50};
    int c[SIZE] = {0};

    // device-side data
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    // allocate device memory
    cudaMalloc((void**)&dev_a, SIZE*sizeof (int));
    cudaMalloc((void**)&dev_b, SIZE*sizeof (int));
    cudaMalloc((void**)&dev_c, SIZE*sizeof (int));

    // copy from host to device
    cudaMemcpy(dev_a, a, SIZE*sizeof (int), cudaMemcpyHostToDevice); // dev_a = a
    cudaMemcpy(dev_b, b, SIZE*sizeof (int), cudaMemcpyHostToDevice); // dev_b = b

    // launch a kernel on the GPU with one thread for each element
    add_kernel<<<1, SIZE>>>(dev_c, dev_a, dev_b); // dev_c = dev_a+dev_b
    cudaDeviceSynchronize(); // kernel launch 지금 수행해!

    // please check CUDA_CHECK() in the common.cpp if you want in detail.
    // cudaPeekAtLastError()
    // - LastError, 마지막 발생한 에러를 Peek, 엿본다는 의미임, 즉, 내가 체크해서 에러를 보고싶다!
    // - cudaPeekAtLastError() 힘수 실행시 발생한 에러가 아니라,
    // - cudaPeekAtLastError() 힘수 이전에 발생한 에러를 체크하는 함수
    cudaError_t err = cudaPeekAtLastError();
    if(cudaSuccess != err)
    {
        printf("CUDA: ERROR: cuda failure \"%s\"\n", cudaGetErrorString(err));
        exit(1);
    }
    else
    {
        printf("CUDA: success\n");
    }


    // copy from deice to host
    cudaMemcpy(c, dev_c, SIZE*sizeof (int), cudaMemcpyDeviceToHost); // c = dev_c

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // print the result
    printf("{%d, %d, %d, %d, %d} + {%d, %d, %d, %d, %d} = {%d, %d, %d, %d, %d}\n",
           a[0], a[1], a[2], a[3], a[4],
           b[0], b[1], b[2], b[3], b[4],
           c[0], c[1], c[2], c[3], c[4]);

    // done
    fflush(stdout); // 화면에 찍어줘!
    return 0;
}

