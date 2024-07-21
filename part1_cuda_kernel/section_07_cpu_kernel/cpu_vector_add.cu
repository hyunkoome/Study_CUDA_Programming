//
// Created by hyunkoo on 24. 7. 16.
//
#include <cuda.h> // bisic header file
#include <cuda_runtime_api.h> // compiler 사용시, runtime function을 사용가능, API = application programming interface (함수 정의)
#include <cuda_runtime.h> // CUDA built-ins, types 등 사용 가능
#include <stdio.h>

void add_kernel(int idx, const int* a, const int* b, int* c)
{
    int i = idx;
    c[i] = a[i] + b[i];
}

int main(void) {
    // host-side data
    const int SIZE = 5;
    const int a[SIZE] = {1, 2, 3, 4, 5};
    const int b[SIZE] = {10, 20, 30, 40, 50};
    int c[SIZE] = {0};

    // calc the addition
    for (register int i = 0; i < SIZE; ++i) {
        //c[i] = a[i] + b[i];
        add_kernel(i, a, b, c);
    }

    printf("{%d, %d, %d, %d, %d} + {%d, %d, %d, %d, %d} = {%d, %d, %d, %d, %d}\n", a[0], a[1], a[2], a[3], a[4], b[0],
           b[1], b[2], b[3], b[4], c[0], c[1], c[2], c[3], c[4]);
    // done
    return 0;
}

