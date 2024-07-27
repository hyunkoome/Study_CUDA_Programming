# Matrix Multiply

***신형 CUDA device 개선점***
- SM: streaming multiprocessor
  - 최신 구조는 `SM 내에 독립 unit` 이 2개, 그 이상 일 수도 있음
    - 1 unit = 32 SP + 16 DP + 8 SFU 로 구성
    - `SP`: streaming processor = core
    - `DP`: double precision
      - 더 정밀도 높은 floating point 숫자 연산 유닛
    - `SFU`: special function unit 
      - cos, sin 같이 복잡한 함수들을 굉장히 빨리 처리할 수 있는. 유닛
  - `Shared Memory 와 L1 cache 포함` 


***Programmer View of CUDA Memories***
- Each thread can:
  - read/write per-thread `registers`
  - read/write per-block `shared memory`
  - read/write per-thread `local memory`
    - actually, located on the `global memory` <== `예전에는 local = global 메모리였는데,`
    - now, `on the L1 cache !` <== `지금은 새로운 CPU에서는, local = L1 cache로..!!`
      - 그러나, L1 cache 가 가득차면, global memory 로 보내서, 용량을 키워서 사용할 수 있게 해줌!
  - read/write per-grid `global memory`
  - `read-only` per-grid `constant memory`


***변수들 실제 스피드 측정 (`speed_variables.cu`)***
- core 1개 사용 => 1개의 쓰래드만 실행해서, 1개의 코어만 사용해서, 스피드 계산
- 1024 x 1024 개의 더하기 = 1M (밀리언) 개 덧셈
- 쉐어드 메모리: 8 x 1024 엘리먼트 = 8K 개의 엘리먼트, 
  - => floating point 사용 => 개당 4 바이트(B) 필요!
  - => 8K x 4 B = 32 KB 용량의 shared 메모리 사용함
  - 대부분, 우리사 사용하는 GPU의 shared 메모리 용량이 최대 48KB 정도이므로,
  - 이 정도 32 KB 정도 잡는게 안전 함
- 연산 시간
  - register 변수:         4,653,833 ticks
  - shared memory 변수:    7,215,134 ticks
  - global memory 변수:   90,934,124 ticks
  - local memory 변수:     7,554,878 ticks
  - constant memory 변수:  4,653,790 ticks 
    - constant 는 읽기용으로만 사용가능해서, register 변수 하나 추가해서 
    - register 변수에 constant 더하는 식으로 ..
    - 그래서, 연산시간은.. 절반은 register 사용한 시간이고, 절반은 constant 변수 사용한 시간으로 봐야 함!
- `의외로, constant 변수를 사용한 경우가 빠름`
  - `read only 만 하는 경우라면, constant 변수를 꼭 사용해야 함!!`
  - `64KB로 제한`이 있긴하지만, ..최대한 활용해라!
- 예상한 대로, `register variable > shared variable > global variable 순서로 빠르다.`
- `예상 외로, local variable 이 상당히 빠름 -> 이제 L1 cache에 위치` (CUDA 6.0 이후)

***Shared Memory 와 L1 Cache***
- 지금은. 쉐어드 메모리와 L1 캐쉬가 1군데의 물리적으로 같은 메모리를 공유하기때문에..
  - 아래 처럼, 두개 합쳐서.. 64 KB 임 
  - shared memory: 프로그래머가 관리하는 영역
  - cache 는 CUDA 가 관리하는 영역임.
  - cache 크기는 shared memory 할당 후 남는 영역이 자동으로.. 잡힘!
- shared memory = `software-managed cache`
  - 정확히는 L1 cache
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
- `__host__ cudaError_t cudaFuncSetCacheConfig (const void* func, cudaFuncCache cacheConfig );` 로 세팅 가능
  - sets the preferred cache configuration for a device function
  - func : device function symbol ( , kernel 함수 이름)
  - cacheConfig : 원하는 cache configuration
    - cudaFuncCachePreferNone: no preference
    - cudaFuncCachePreferShared: shared memory is 48 KB
    - cudaFuncCachePreferEqual: shared memory is 32 KB
    - cudaFuncCachePreferL1: shared memory is 16 KB -> L1 캐시 용량이 48KB가 됨 

***추가 고려 사항들***
- Any memory allocation function does NOT guarantee to clean the memory
  - 원래는, `메모리 할당 함수`(malloc( ), cudaMalloc( ), cudaMallocPitch( ), cudaMalloc3D( ), …)는 `변수 값 초기화 시켜주지는 않음.`
  - OS에 따라, clear 되는 경우도, 있긴 함.
- 많은 CPU/CPU에서 memory clear 기능을 별도로 제공
- memset( ): `assembly 언어 레벨에서 최적화`된 routine
  - intel cpu의 경우, SSE extension 적용
  - window OS의 ZeroMemory(): memset()과 동일
- `Memory Set Functions`: 메모리를 특정한 값으로 set
  - #include <string.h>
  - void* `memset`( void* ptr, int value, size_t count );
  - __host__ cudaError_t `cudaMemset` ( void* devPtr, int value, size_t count );
  - __host__ cudaError_t `cudaMemset2D` ( void* devPtr, size_t `pitch`, int value, size_t width, size_t height );
  - __host__ cudaError_t `cudaMemset3D` ( cudaPitchedPtr `pitchedDevPtr`, int value, cudaExtent extent );
    - initializes or sets `each byte` in the memory to a value `단위는 바이트 임!!`
    - ptr, devPtr : start address to the destination
    - value : value to set for each byte
  
- memory clear (보통 global 메모리)시 속도: `32 M(백만) 개 메모리 0로 리셋`
  - cpu
    - 사용자 함수 `clear with for-loop`: 14,141 usec
    - `memset`: 8,650 usec 
  - cuda 
    - 사용자 함수 `kernelClear()`: 249 usec
    - `cudaMemset`: 149 usec 

[Return Par4 Matrix Multiply](../README.md)  