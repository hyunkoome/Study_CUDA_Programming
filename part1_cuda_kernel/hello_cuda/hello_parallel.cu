//
// Created by hyunkoo on 24. 7. 14.
#include <stdio.h>

// cuda
__global__ void hello_cuda() // <-- cuda: add __global__
{
    printf("Hello, CUDA!: %d\n", threadIdx.x); // 실행되는 core 번호: threadIdx.x
}

int main() {
    // <-- cuda: add <<<1,8>>>, 1세트 x 8번씩 동시 실행 -> 즉, 8개 cuda core 사용해서 실행
    // this function run 8 times
    hello_cuda<<<1,8>>>();
#if defined(__linux__)
    cudaDeviceSynchronize(); // 그래픽카드가 하고있던 모든일을 완료해라. 즉, 이 함수도 지금 실행해라는 의미임.
#endif
    fflush(stdout);

    // 8세트 x 2번씩 동시 실행 -> 즉, 16개 cuda core 사용해서 실행
    // threadIdx.x 가 0, 1로 된 세트가 8개 돌아간다는 의미
    hello_cuda<<<8,2>>>();
#if defined(__linux__)
    cudaDeviceSynchronize(); // 그래픽카드가 하고있던 모든일을 완료해라. 즉, 이 함수도 지금 실행해라는 의미임.
#endif
    fflush(stdout);

    return 0;
}

// not run on the linux