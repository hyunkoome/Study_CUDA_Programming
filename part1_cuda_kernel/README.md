# Study CUDA_Programming (based on C++ ver11)

**header 관련해서**
- nvcc compiler 사용시 아래 3개 header는 자동으로 적용되는 경우가 많지만, 추가해주도록 !!
```c++
#include <cuda.h> // bisic header file
#include <cuda_runtime_api.h> // compiler 사용시, runtime function을 사용가능, API = application programming interface (함수 정의)
#include <cuda_runtime.h> // CUDA built-ins, types 등 사용 가능
```
- not CUDA standard header file, provided by CUDA samples,  you can find them in CUDA samples / inc directory
```c++
#include <helper_cuda.h>
#include <helper_functions.h>
```
- 모든 CUDA 함수는 cuda로 시작
- 대부분 모든 CUDA 함수는 에러 코드를 리턴, (성공 시는 cudaSuccess 를 리턴)
- Please check [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
  - nvcc: -gencode 
    - GeForce RTX 4090: Compute Capability - 8.9
    - –gencode=arch=compute_89,code=\"sm_89,compute_89\"
      –arch=sm_89
  - for more details, read the NVCC manual
  - set it in your “makefile” (or other equivalents)


**Easy way to compile**
```shell
nvcc -o hello_cuda hello_cuda.cu -DCMAKE_CUDA_ARCHITECTURES='89'
```
- here: -O2 (오투) : build release mode
```shell
nvcc --gpu-architecture=compute_89 --gpu-code=sm_89,compute_89 -O2 -o hello_linux hello_linux.cu
```
- add architecture
```shell
/usr/local/cuda/bin/nvcc -ccbin g++ -I../../../Common  -m64 --std=c++14 --threads 1 -gencode arch=compute_89,code=sm_89 -o segmentationTree.o -c segmentationTree.cu
```

**build**
- In each folder
```shell
mkdir build
cd build
cmake ..
make
```

**CUDA programming model**
- use PREFIX for each function
  - `__host__`
    - can be called by CPU (default, can be omitted)
    - cpu (host) 에서 사용, 생략 가능
    - CPU 가 처리
  - `__device__`
    - called from other GPU functions, cannot be called by the CPU
    - GPU에서 사용, cpu에서 사용 X
    - CUDA 가 처리
  - `__global__`
    - launched by CPU, cannot be called from GPU, must return void
    - GPU에서 사용은 가능하지만, 대신, 실행을 launch 하는 것, 즉, 실행하도록 call 해주는 것은 CPU가 하는 함수가 됨
    - CUDA 가 처리
    - kernal fuction 이라고 부름, cuda 에서 젤 많이 사용함
      - cpu 가 cuda 에 이제 니가 일하라고 지시할때(host->device) call 하는 함수가 global 함수 임
  - `__host__` and `__device__` qualifiers can be combined
    - 경우에 따라서, 나중에 복잡한 프로그램을 짜다보면, 하나의 function 을 cpu도 사용하고, cuda도 사용해야 할때,
    - `__host__` `__device__` 둘다 붙이면, 컴파일러가 두번(cpu, cuda) 컴파일 함.
- C/C++ 언어와의 차이 
  - CUDA: GPU 메모리만 접근 가능
    - 새로운 버전에서는 가능하지만, 사용하지 마세요!
  - static 변수 X
  - recursion 재귀 호출 X
    - 새로운 버전에서는 가능하지만, 사용하지 마세요!
  - dynamic polymorphism 다중 상속 X
- 커널 kernel function
  - loof body를 따로 때냈을때, 이 함수를 .., 즉, loof 의 body 부분만 함수로 구현한 것
- CUDA 에러 처리
  - CUDA 시스템은 내부적으로 error flag 를 하나 갖고 있음: 한개 에러만 저장 
  - cudaError_t: CUDA 에러에 대한 데이터 타입  
  - cudaError_t cudaGetLastError(void)
    - 에러를 갖고 오면, Error flag를 리셋, 다음 에러를 또 받을수 있도록 준비
    - 내가 에러 처리를 직접 할께, 즉, 에러처리라고 함은, 에러 print 하고, Error flag를 리셋    
  - cudaError_t cudaPeekAtLastError(void)
    - 에러를 갖고 와도, Error flag를 리셋 X
    - 따라서, 같은 에러를 한번 체크만 하고, 실제 처리는 누군가 딴 애가 해야될 때는 이 함수를 사용
    - 스스로 에러 처리는 못함
  - 에러 관련 함수들
    - cudaGetErrorName, cudaGetErrorString
    - cudaGetLastError, cudaPeekAtLastError
    - CUDA error check macros
  


