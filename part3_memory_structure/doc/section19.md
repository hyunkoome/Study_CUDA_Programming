# Memory Structure

### memory_hierarchy 메모리 계층 구조 

***Shared Memory in CUDA***
- register 아니고 memory 다
  - 그래서, 배열 잡을수 있다.
  - 그러나, register 만큼 빠르다.
  - 용량도 큼
  - 도대체 뭔데?? Shared Memory = cache 임
- Scratchpad Memory (SPRAM)
  - cache memory to support real-time calculations
    - CPU에서는 사용자가 직접 제어 & 사용 못함
    - 그러나 `CUDA에서는 shared memory로 사용 가능하게 되어 있음`
- 따라서, 프로그래머가 Shared Memory 를 잘 쓰면, 괭장히 빠른 프로그램을 만들수 있음

***CUDA variable type qualifiers***
- `int var;` 
  - `CUDA 커널 프로그램 내에서, 아무런 표시를 않하면 register memory 변수 임,` 
  - 속도 1x (`제일 빠름`), `one clock 에 처리`
  - 대신에, 레지스터에 있는 변수들은 thread 가 exclusive 하게 사용함
    - 즉, '배타적'으로 사용하니까, thread 만 사용할 수 있음
    - 즉, `다른 thread 에서는 사용 못하고, 각자의 thread 내에서만 독립적으로 사용` 
  - [`※ 주의 점`]
    - 특별히 말하지 않으면 레지스터로 가는게 일반적인데,
    - 사용 할 수 있는 레지스터가 더 이상 없으면,
    - 그때는 자동으로 local memory로 감 => global memory로 갈 가능성이 아주 큼. 굉장히 느려짐
    - 그래서, CUDA 프로그래머 들은, 레지스터의 개수에 괭장히 민감함.
- `int array[10];`
  - 경우에 따라서, `우리가 굳이 배열이 필요하다면,`
  - 커널 프로그램 내에서 배열을 잡으면 => 그것은 `local memory 로 감`
    - 예전 CUDA 는 global 메모리로 보냈는데,
    - 다행히, 요즘의 CUDA 는 이것을 shared memory, 즉, cache로 보기도 함
    - 그래도 local memory 는 `상당히 느리다` 라고 봐야 함
    - 즉, `register 보다 100배 정도 느리다.`
- `__shared__ int shared_var;`
  - shared memory 변수
  - 속도 면에서는 레지스터 만큼 빠름
  - 같은 `thread block` 에 있는 애들끼리만 `공유` 가능 
    - 물론 thread는 당연히 사용
    - 추가로, 같은 `thread block` 에 있는 애들끼리 서로 정보를 공유하는게 가능함 
  - `__shared__` 앞에 `__device__` 붙여도 되고 안붙여도 됨
- `cudaMalloc( &dev_ptr, size );`
  - cudaMalloc 으로 만들면, global memory 사용
  - 속도면에서는 레지스터에 비하면 100배 이상 느림
  - grid에 잡힘
    - 같은 grid에 있는 thread 들, 즉, 커널 launch 했을때 돌아가는 모든 thread가
    - 같은 global 메모리를 access 하니까, 서로 데이터를 공유하는게 가능함
- `__device__ int global_var;`
  - global memory 사용
  - ※ ch) `__global__` 는 특별히 커널 function 용어로 사용했음
- `__constant__ int constant_var;`
  - `__constant__` 는 C/C++에서의 constance 변수 즉, 값 못바꾸는 변수의 의미가 아니라
    - CUDA의 constant memory로 간다는 의미
    - constant memory 는 대부분 읽기만 하고, 쓰지는 않는다는 가정을 해서, 
      - cache memory 부분에 갖다 놓는 바람에 속도가 상당히 빠름
  - grid에 잡힘
    - 같은 grid에 있는 thread 들, 즉, 커널 launch 했을때 돌아가는 모든 thread가
    - 같은 global 메모리를 access 하니까, 서로 데이터를 공유하는게 가능함
  - `__constant__` 앞에 `__device__` 붙여도 되고 안붙여도 됨

***CUDA variable 의 선언 위치***
- 커널 함수 외부: host에서 memcpy 가능
  - global variables
  - cudaMalloc 로 메모리 영역을 잡거나
  - constant variables: constant 변수 잡거나.
- 커널 `함수 내부` : host 와 연결 불가능
  - `register variables`
  - `local variables` (or thread-local variables)
  - `shared memory variables`
  - 위 3개의 변수들은 main memory 하고 직접 연결 되지 못함
    - 즉, CPU와 직접 통신은 불가능 
    - 대신, 하나의 thread block 안에서, 서로 데이터를 주고받거나 하는 것까지는 가능하다고 보면 됨 => shared memory
    - 레지스터나 local variable은 한 thread 만 사용할 수 있다고 봐야함

***실습 코드***
- AXPY 문제 풀기
  - `saxpy_cuda_fma.cu`: cudaMalloc 사용 
    - 실행 시간: 996 usec
  - `saxpy_symbol.cu`: 커널 변수 사용
    - 코드가 아주 간단해 짐
    - 실행 시간: 960 usec
    - const unsigned vecSize = 256 * 1024 * 1024; // 64 -> 256 으로 more big-size elements 바꾸면
      - 아래와 같이 에러 발생하고 컴파일이 안됨 
      - Windows:
        - LINK : fatal error LNK1248: 이미지 크기(80071000)가 허용 가능한 최대 크기(80000000)를 초과 합니다.
      - Linux:
        - /usr/bin/ld: failed to convert GOTPCREL relocation; relink with --no-relax
        - collect2: error: ld returned 1 exit status
      - `컴파일러 이슈로 인해. static 변수들을 일정 크기 이상의 변수를 잡을 수 없음`!!!
        - 그래서. Malloc 을 사용할 수 밖에 없음.

***cudaGetSymbolAddress( )***
- `__host__ cudaError_t cudaGetSymbolAddress (void** devPtr, const void* symbol );`
  - finds the address associated with a CUDA symbol.
  - devPtr : return device pointer associated with symbol
  - symbol : device symbol address (__device__로 선언된 array 이름)
- 결과적으로, cudaMemcpyToSymbol() / cudaMemcpyFromSymbol() 대신, 아래와 같이 코드 수정 가능하여,
  - `cudaGetSymbolAddress` 사용하면, 기존의 `cudaMemcpy` 함수 사용할 수 있음   
```c
void* ptr_x = nullptr;
cudaGetSymbolAddress( &ptr_x, dev_x );
cudaMemcpy( ptr_x, host_x, vecSize * sizeof(float), cudaMemcpyHostToDevice );
```



[Return Par3 Memory Structure](../README.md)  