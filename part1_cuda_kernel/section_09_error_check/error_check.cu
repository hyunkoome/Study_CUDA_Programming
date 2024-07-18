//
// Created by hyunkoo on 24. 7. 16.
//

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "stdio.h"
#include "./common.cpp"

#if defined(NDEBUG) // release mode
#define CUDA_CHECK_ERROR_KIM() 0 // 0으로 선언하면, 컴파일러가 해당 메크로를 없다고 무시하기때문에, release에서는 적용안하는것이 되버림
#else // debug mode
// __FILE__: 컴파일러가 현재 파일 이름을 넣어줌
// __LINE__: 컴파일러가 현재 라인 넘버, 몇번째 줄 인지 넣어줌
#define CUDA_CHECK_ERROR_KIM()  do { \
        cudaError_t e = cudaGetLastError(); \
        if (cudaSuccess != e) { \
            printf("error check kim: cuda failure \"%s\" at %s:%d\n", \
                   cudaGetErrorString(e), \
                   __FILE__, __LINE__);  \
            exit(1); \
        }                        \
        else{                    \
              printf("error check kim: CUDA: success\n"); \
        }                         \
    } while (0)
#endif


__global__ void add_kernel(float *b, const float *a) {
    int i = threadIdx.x;
    b[i] = a[i] + 1.0f;
}

int main() {
    const int SIZE = 8;
    const float a[SIZE] = {1., 2., 3., 4., 5., 6., 7., 8.};
    float b[SIZE] = {0.,};

    // print source
    printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    fflush( stdout );

    float* dev_a = nullptr;
    float* dev_b = nullptr;

    cudaMalloc((void**)&dev_a, SIZE*sizeof (float ));
    cudaMalloc((void**)&dev_b, SIZE*sizeof (float ));

    cudaMemcpy(dev_a, a, SIZE*sizeof (float ), cudaMemcpyHostToDevice); // dev_a = a

    // kernel
    add_kernel<<<1, SIZE>>>(dev_b, dev_a);
    cudaDeviceSynchronize();

    // print destination
    cudaMemcpy(b, dev_b, SIZE*sizeof (float ), cudaMemcpyDeviceToHost); // b = dev_b
    printf("{%f, %f, %f, %f, %f, %f, %f, %f}\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
    fflush( stdout );
    CUDA_CHECK_ERROR_KIM();

    // error check
//    cudaError_t err = cudaGetLastError();
//    if(cudaSuccess != err)
//    {
//        printf("CUDA: ERROR: cuda failure \"%s\"\n", cudaGetErrorString(err));
//        exit(1);
//    }
//    else
//    {
//        printf("CUDA: success\n");
//    }

    // free cuda memory
    cudaFree(dev_a);
    cudaFree(dev_b);

    // error check
    CUDA_CHECK_ERROR();

    // done
    fflush(stdout);
    return 0;
}