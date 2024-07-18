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
    //cudaDeviceSynchronize(); // 그래픽카드가 하고있던 모든일을 완료해라. 즉, 이 함수도 지금 실행해라는 의미임.

    /* 버퍼에 데이터가 남게 되면 정상적인 입출력을 하지 못하기 때문에, fflush()함수를 이용하여 버퍼에 있는 데이터를 비워줘야 합니다.
    1. 입력 스트림(stdin)
       입력 버퍼 안에 존재하는 데이터를 비우는 즉시 삭제합니다.
    2. 출력 스트림(stdout)
       출력 버퍼 안에 존재하는 데이터를 비우는 즉시 출력합니다.
       (버퍼에 있는 데이터를 비운다는 게 헷갈리신다면, 버퍼에 있는 데이터를 꺼내 출력장치로 보낸다고 생각하시면 됩니다.) */
    fflush(stdout);
    // done

    return 0;
}