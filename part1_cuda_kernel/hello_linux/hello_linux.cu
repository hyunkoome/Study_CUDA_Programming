#include <cuda.h> // bisic header file
#include <cuda_runtime_api.h> // compiler 사용시, runtime function을 사용가능, API = application programming interface (함수 정의)
#include <cuda_runtime.h> // CUDA built-ins, types 등 사용 가능
#include <stdio.h>

__global__ void hello_cuda() // <-- cuda: add __global__
{
    printf("Hello, cuda linux!\n");
}

int main() {
    hello_cuda<<<1,1>>>(); // <-- cuda: add <<<1,1>>>, it means 1x1 -> use 1ea cuda core
    cudaDeviceSynchronize(); // 그래픽카드가 하고있던 모든일을 완료해라. 즉, 이 함수도 지금 실행해라는 의미임.
    return 0;
}