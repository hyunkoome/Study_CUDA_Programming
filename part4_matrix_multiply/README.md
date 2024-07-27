# Matrix Multiply 행렬 곱셈

[Section22: matrix copy](./doc/section22.md)
- CPU 사용
  - CPU에서 이중 루프 돌린거 보다 메모리 카피 쓴게 의외로 약간 느림
  - 이 결과는 메모리 카피가 운영체제에서 지원 되면서, 굉장히 효과적으로 짜여있긴하지만,
  - 경우에 따라서는 cashe 상황이나, 혹은 컴파일러가 for 루프를 얼마나 optimize 했느냐에 따라,
  - 이런 관계 때문에, `경우에 따라서는 우리가 for 루프 돌려서 직접 짠 것이 더 빠를수도 있다는 것을 보여줌`
- CUDA naive version – `global memory`
  - naive : (전문적) 지식이 없는
  - `pitched matrix` 를 사용하는 것이 속도가 빠르므로.. 이 방법을 사용
  - 분명히, cpu 버전들 보다 빨라짐
- CUDA `shared memory` 사용: `tiled approach` 사용 해야함
  - 글로벌 메모리 사용 버전과 비교하면, 다소 느리지만,
  - 그 이유는 shared 메모리를 한번 더 거쳤으니까, 느릴수 밖에 없다.
  - `그러나, 일반적으로 shared 메모리에서 행렬 copy가 아닌, 실제 연산등이 추가되므로.`
  - `이 방법을 사용하는 것을 연습하기를 권장 함`
- CUDA memcpy2D 쿠다 커널 직접 카피 (`cudaMemcpy2D()`)
  - CUDA에서 글로벌 메모리 쓰는 버전이나, 쉐어드 메모리 쓰는 버전보다. 빨라짐
  - 그래서, `사실은 이것이.. 우리가 CUDA 프로그래밍할때 기준이 되는, 속도 레퍼런스가 됨`
    - 이것 보다 더 빠른 알고리즘을 우리가 CUDA에 구현하는 것은 불가능함.
  - the best score 최고 기록 for any matrix operations 행렬 대입 연산이 최고 기록이다.
    - 따라서 다른 연산, 즉 더하기 연산 등은 이 행렬 대입연산의 속도에 근접할 수록 최적화가 잘 된것이고
    - 이 기록을 추월했다면, 만세를 부를게 아니라, 뭔가 잘 못된 것임
    - 즉, 연산의 물체를 발견할 수 있는 어떤 기준이 될수도 있음

[Section23: Matrix Transpose 전치 행렬](./doc/section23.md)
- Matrix Transpose 전치 행렬
  - C = A^T = A^tr
    - C[i,j] = A[j,i]
    - for simplicity, we assume `square matrices`
    - A[y][x] = A[y * WIDTH + x]
      - => C[y][x] = A[x][y] = A[x * WIDTH + y]
- 실습 구현
  - CPU version
  - CUDA naive version – global memory
    - 두가지 matrix 모두 pitched 된거 고려해서 구현 필요! (글로벌 index 사용)
  - CUDA shared memory, naive version
    - tiled approach 사용 (글로벌 index, local index 사용)
  - CUDA shared memory, optimized version
    - (행단위로 ) 가로로 저장을 해야, 속도가 빠르므로, 아래와 같은 트릭을 사용
      - 읽어오는 부분은 M[y][x] = A[y][x]
      - 저장하는 부분은 C[y][x] = M[x][y]
  - CUDA shared memory, `bank conflict` resolved version 메모리 뱅크 출동 현상
    - 쉐어드 메모리의 특성을 잘 이용하는 방법: `back conflict 상황은 피하게!!`
      - 세어드 메모리는 물리적으로 32개의 bank로 구성 (SP가 32개 이므로.)
      - 4개씩 끊어서, 4 bytes 씩 끊어서, 이 back 에 쫙~ 넣어놨음
      - bank 단위로 독립적으로 메모리를 access 할 수 있음
      - 그래서, back confict 가 없는 경우는. 가장 빠름.
        - 32개 데이터를 32개 뱅크에서 1개씩 나눠서 갖고있어서
          - 32개 SP에서 1개씩 엑세스해서 처리하니, 1싸이클에 끝남
      - 그러나, back confict 가 있는 최악 의 경우는. 가장 느림.
        - 32개 데이터를 1개 뱅크에서 모두 갖고 있어서.
        - 1개 SP에서 32번 처리해야 함. 32 싸이클을 거쳐야 끝남

[Section24: Matrix Multiplication](./doc/section24.md)
- C = A x B
  - C[i][j] = dot product `내적` of A[i][_] and B[_][j]
  - C[i][j] = sum (A[i][k] x B[k][j]) from k=0 to k=n-1
  - `각 행렬 원소 (y,x) 마다 for loop 계산`
    ```c++
    register float sum = 0.0f;
    for (unsigned k = 0; k < WIDTH; ++k) {
    sum += a[i][k] * b[k][j];
    }
    c[i][j] = sum;
    ```
    - => `3중 for loop`
- 3중 for loop의 개선책
  - outer-k version `3중 for-loop 문도, 순서를 바꾸면 빨라짐!`
    - 전체 계산하는 양은 똑같은데, 캐시를 잘 쓸 수 있어서, 속도가 빨라짐
    - 실행 시간: 92,237,678 usec = 92초 = 1분 32초
- CUDA implementation
  - global memory
    - grid 설계: 2D layout
      - matrix C 의 각 원소 (y,x) 마다 thread (block) 생성
      - 각 thread 는 for loop 로 row x column 계산
    - 3중 for loop 에서 바깥쪽 2중 for loof 는 `2D grid (block thread) 구조`로 대체
      - 안쪽의 하나의 for 루프만 커널 내에서 실행 하도록..
    - 이 모든 메모리 구조가 다 `pitched` 되어있다는 것은 신경써야 함!!
  - shared memory - tiled approach
    - warp 단위로 처리 중
      - `읽어올때`, `계산할때`, 각각 `다른 트릭`을 사용함!
        - 1) tile 을 갖올때는 row-major (옆줄 단위로) 로 가져오는 게 빠름!
        - 2) 싱크 쓰래드 사용해서 동기화!
        - 3) 계산 processing 은 각각 row, column 으로
    - tile 크기를 32 x 32 로 assume
      - matrix 크기를 32의 배수로 assume (4k x 4k 사용)

[Section25: GEMM: general matrix-to-matrix multiplication](./doc/section25.md)
- general matrix-to-matrix multiplication
  - Z = alpha A B + beta C
    - A, B, C, Z: matrices
    - alpha, beta: scalar values
  - `핵심`은 `행렬 곱하기`를 어떻게하면 빠르게 하느냐 이다.
    - 행렬 곱하기 강의 (`Section 24`를 복습!!)
- BLAS level 3 operation
  - BLAS = Basic Linear Algebra 기능들(Subprograms)
  - BLAS 함수 중에서 가장 많이 쓰이고.
  - BLAS optimization 최적화에서 가장 중요하게 고려
- implementations
  - 3중 for-loop
  - for-loop 순서 변경 -> k를 제일 밖으로 -> 당연히, 순서 바꾸면 `10배 정도는 빨라져야 함!!`
  - cuda global 메모리 사용
    - grid 설계: 2D layout
      - matrix C 의 각 원소 (y,x) 마다 thread (block) 생성
      - 각 thread 는 for loop 로 row x column 계산
      - 3중 for loop 에서 바깥쪽 2중 for loof 는 `2D grid (block thread) 구조`로 대체
        - 안쪽의 하나의 for 루프만 커널 내에서 실행 하도록..
      - 이 모든 메모리 구조가 다 `pitched` 되어있다는 것은 신경써야 함!!
  - CUDA aligned tile 
    - tile 크기를 32x32 로 가정
    - `matrix 크기 를 32의 배수로 가정`
- 속도는 !!
  - CUDA aligned tile 사용하는 경우가 가장 빠름.
  - 행렬의 크기를 4,096 같이 32의 배수가 되도록 딱 맞춰주면, 가장 빨리 계산할 수 있다.!

[Section26: 메모리에 따른 CUDA 변수 스피드 측정](./doc/section22.md)
- 신형 CUDA device 개선점
  - SM: streaming multiprocessor
    - 최신 구조는 `SM 내에 독립 unit` 이 2개, 그 이상 일 수도 있음
      - 1 unit = 32 SP + 16 DP + 8 SFU 로 구성
      - `SP`: streaming processor = core
      - `DP`: double precision
        - 더 정밀도 높은 floating point 숫자 연산 유닛
      - `SFU`: special function unit
        - cos, sin 같이 복잡한 함수들을 굉장히 빨리 처리할 수 있는. 유닛
    - `Shared Memory 와 L1 cache 포함`
- Programmer View of CUDA Memories
  - Each thread can:
    - read/write per-thread `registers`
    - read/write per-block `shared memory`
    - read/write per-thread `local memory`
      - actually, located on the `global memory` <== `예전에는 local = global 메모리였는데,`
      - now, `on the L1 cache !` <== `지금은 새로운 CPU에서는, local = L1 cache로..!!`
        - 그러나, L1 cache 가 가득차면, global memory 로 보내서, 용량을 키워서 사용할 수 있게 해줌!
    - read/write per-grid `global memory`
    - `read-only` per-grid `constant memory`
- 변수들 실제 스피드 측정
  - `의외로, constant 변수를 사용한 경우가 빠름`
    - `read only 만 하는 경우라면, constant 변수를 꼭 사용해야 함!!`
    - `64KB로 제한`이 있긴하지만, ..최대한 활용해라!
  - 예상한 대로, `register variable > shared variable > global variable 순서로 빠르다.`
  - `예상 외로, local variable 이 상당히 빠름 -> 이제 L1 cache에 위치` (CUDA 6.0 이후)
- Shared Memory 와 L1 Cache
  - 지금은. 쉐어드 메모리와 L1 캐쉬가 1군데의 물리적으로 같은 메모리를 공유하기때문에..
    - 아래 처럼, 두개 합쳐서.. 64 KB 임
    - shared memory: 프로그래머가 관리하는 영역
    - cache 는 CUDA 가 관리하는 영역임.
    - cache 크기는 shared memory 할당 후 남는 영역이 자동으로.. 잡힘!
  - 최근 CUDA device의 설정 (한 군데로, 몰수는 없음!!)
  - `shared memory + L1 cache = 64K Byte`
  - 용량은 flexible
    - shared memory 48KB + L1 cache 16KB <- `디폴트`
    - shared memory 32KB + L1 cache 32KB
    - shared memory 16KB + L1 cache 48KB
  - kernel 속도는?
    - 개발자 입장에서는, `shared memory를 많이 쓰면,`
      - 내 프로그램이 빨라진다고 느끼기가 쉬운데,
      - 대신에, L1 cache 용량이 축소 됨 => L1 cache 가 줄면, 프로그램이 느려질수도 있음
    - => 실제 영향은 측정 필요!
- 추가 고려 사항들
  - Any memory allocation function does NOT guarantee to clean the memory
    - 원래는, `메모리 할당 함수`(malloc( ), cudaMalloc( ), cudaMallocPitch( ), cudaMalloc3D( ), …)는 `변수 값 초기화 시켜주지는 않음.`
    - OS에 따라, clear 되는 경우도, 있긴 함.
    - 많은 CPU/CPU에서 memory clear 기능을 별도로 제공
    - memset( ): `assembly 언어 레벨에서 최적화`된 routine
  - `Memory Set Functions`: 메모리를 특정한 값으로 set
    - #include <string.h>
    - void* `memset`( void* ptr, int value, size_t count );
    - __host__ cudaError_t `cudaMemset` ( void* devPtr, int value, size_t count );
    - __host__ cudaError_t `cudaMemset2D` ( void* devPtr, size_t `pitch`, int value, size_t width, size_t height );
    - __host__ cudaError_t `cudaMemset3D` ( cudaPitchedPtr `pitchedDevPtr`, int value, cudaExtent extent );
      - initializes or sets `each byte` in the memory to a value `단위는 바이트 임!!`
      - ptr, devPtr : start address to the destination
      - value : value to set for each byte
  
[Section27: 정밀도와 속도개선](./doc/section27.md)
- floating point numbers 실수 표현 방법
  - IEEE 754 floating-point standard `부동 소수점 방식`
    - `float`, `double`, `long double`
- 정밀도 문제
  - algorithmic consideration
    - `(small + small) + large → may be more accurate`
    - (large + small) + small → pot
    - 아주 작은 숫자를 큰숫자에 더하면, 아주 작은 숫자가 사라짐
      - 왜냐.. `소수점 7자리까지 밖에 저장`을 못하므로.
      - 그래서, `아주 작은 숫자들 끼리 먼저 더해서.. 큰 숫자에 더하는 것이..`
      - `계산 오차를 줄여줌!!`
- float-safe 최적화
  - `float x = 1.0f;`
    - `1` 로 적으면, int로  잡혀서, 형 변환이 필요하여, 추가 clock 필요!
    - 그래서, `f`붙이는 습관!!!
  - float y = 1.0f * x;
    - `1.0`이면 double로 잡혀서, 형 변환이 필요하여, 추가 clock 필요!
    - 그래서, `f`붙이는 습관!!!
  - float z = 3.0f * sinf( y );
    - 일반 수학 함수는 double 형을 위한 것으로..
    - `float 형 을 위한 함수는 보통 뒤에. f가 붙음`
      - 속도가 조금이라도 빨라짐
- CUDA Runtime MATH library
  - `func( )` : compile to ***multiple instructions*** (or library functions)
    - `slower but higher accuracy` 정밀도가 목적..
    - examples: sin(x), sinf(x), exp(x), expf(x), pow(x, y), powf(x, y)
  - `__func( )` : direct mapping to a ***single hardware instruction***
    - `fast but low accuracy` 속도가 목적..
    - examples: __sin(x), __sinf(x), __exp(x), __expf(x), __pow(x, y)
  - `–use_fast_math` ***compiler option***:
    - forces every `func( )` to compile to `__func( )`
    - 컴파일 시, nvcc 에, `–use_fast_math` 를 추가하면, 우리가 사용한 모든 수학 함수를 __가 붙은 함수로 바꿔서 컴파일 시켜버림!!
      - 디버그 시에는 일반적인 함수를 사용해서 개발하다가,
      - release 시에는 이 옵션을 붙여서.. 컴파일 하면,,
      - 소스코드를 수정하지 않고도.. 속도가 엄청 빨라짐..
      - 그러나, 정밀도가 떨어질 수도 있으니.. 테스트 해가면서..하는게 제일 안전 함!
- 속도 비교
- `sine / cosine` calculation (with a `single core`, not many-core)
  - CPU version: 16,915 usec
  - CUDA default: 134,234 usec
    - 1개의 core만 썼으므로, cpu 보다 느릴수 밖에 없음.
  - CUDA fast-math (빠르지만, 근사값): 10,109 usec
    - 근사값이라고 하지만, 크게 오차가 나오지 않음.
    - 그래서, 많이 쓰자!!
  - CUDA sincosf( ): 11,873 usec
    - fast-math 보다 더 빨라진다고 하는데.. 실제로는 안빨라지는 듯함.
    - 근데, 강의에서는 빨라지는 듯..
    - 그래서, 나중에 사용할때,,확인이 필요할 것 같음
  - CUDA fma( ): 10,396 usec
    - fma: fused multiply-add instruction (z <- a x + y 문제)
  
[Return Main Readme](../README.md)  


