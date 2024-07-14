//#include <iostream>
#include <stdio.h>

__global__ void hello_cuda() // <-- cuda: add __global__
{
    printf("Hello, cuda!\n");
    // c++ stream 은 cuda에서 지원하지 않음 #include <iostream>
    // std::cout << "Hello, cuda!" << std::endl;
}

int main() {
    hello_cuda<<<1,1>>>(); // <-- cuda: add <<<1,1>>>, it means 1x1 -> use 1ea cuda core
#if defined(__linux__) // 이 블럭 없으면, 리눅스에서 동작하지 않음
    cudaDeviceSynchronize(); // 그래픽카드가 하고있던 모든일을 완료해라. 즉, 이 함수도 지금 실행해라는 의미임.
#endif
    fflush(stdout);
    return 0;
}