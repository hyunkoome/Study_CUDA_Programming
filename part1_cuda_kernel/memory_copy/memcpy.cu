#include <cuda.h> // bisic header file
#include <cuda_runtime_api.h> // compiler 사용시, runtime function을 사용가능, API = application programming interface (함수 정의)
#include <cuda_runtime.h> // CUDA built-ins, types 등 사용 가능
#include <stdio.h>


int main(void ) {
    // host-side data
    const int SIZE = 8;
    const float a[SIZE] = {1.,2.,3.,4.,5.,6.,7.,8.};
    float b[SIZE] = {0.,0.,0.,0.,0.,0.,0.,0.};

    // print source
    printf("a={%f, %f, %f, %f, %f, %f, %f, %f}\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    printf("b={%f, %f, %f, %f, %f, %f, %f, %f}\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
    cudaDeviceSynchronize(); // 그래픽카드가 하고있던 모든일을 완료해라. 즉, 이 함수도 지금 실행해라는 의미임.

    // device-side data
    float* dev_a = nullptr;
    float* dev_b = nullptr;

    // allocate device memory
    cudaMalloc((void**)&dev_a, SIZE*sizeof(float ));
    cudaMalloc((void**)&dev_b, SIZE*sizeof(float ));

    // 3 copies
    cudaMemcpy(dev_a, a, SIZE*sizeof(float ), cudaMemcpyHostToDevice); // dev_a = a
    cudaMemcpy(dev_b, dev_a, SIZE*sizeof(float ), cudaMemcpyDeviceToDevice); // dev_b = dev_b
    cudaMemcpy(b, dev_b, SIZE*sizeof(float ), cudaMemcpyDeviceToHost); // b = dev_b

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);

    // print the result
    printf("b={%f, %f, %f, %f, %f, %f, %f, %f}\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
    cudaDeviceSynchronize(); // 그래픽카드가 하고있던 모든일을 완료해라. 즉, 이 함수도 지금 실행해라는 의미임.
    fflush(stdout);
    // done

    return 0;
}