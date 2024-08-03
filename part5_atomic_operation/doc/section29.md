# atomic operation

***race conditions***
- 병렬처리에서 여러 threads 에서 하나의 변수를 서로 읽거나, 업데이트 하고 싶어할때 발생
  - shared 메모리, global 메모리 모두 대상 임
- global communication
  - 서로 다른 스래드 블록 간의 통신
  - 글로벌 메모리를 대상으로..

***race conditions 해결잭***
- memory locking
  - 전통적인 해결책
  - lock - update - unlock
  - lock 실패시, 기다림
- atomic operation
  - 새로운 해결책 => CUDA 도 채택
  - atomic update (a single operation)
    - 어셈블리 명령어로.. 1싸이클에 완료됨.
  - 동시 시도 가능. 1개만 성공

***atomic operations***
- The operation is atomic.
  - hardware 에서 지원하는, `single step 연산`
- `인터럽트 불가능` uninterruptable
  - sub-step 으로 분할되지 않고, `한꺼번에 수행`
  - 여러개의 thread 가 동시에 atomic operation 수행 가능
  - `단 1개의 thread 만 성공`, 나머지는 모두 실패,,
- 다양한 연산 제공
  - atomic { Add `+`, Sub `-`, Exch `교환`, Min, Max, Inc `++`, Dec `--`, `CAS`, And `&`, Or `|`, Xor `^`}
  - CAS: `compare and swap`
    - the `most fundamental operation` among all atomic operations 아토믹 연산의 기본
  - `int` atomicCAS(int* `address`, int `old_value`, int `new_value`);
    - step 1. current = `read`(*address); 
      - `현재(current) 값을 체크합니다.`
    - step 2. (*address) = `(current == old_value) ?` new_value : current; 
      - `이 값이 옛날 값이면, 새로운 값으로 업데이트 하고, 새 값이면, 그대로 둠`
    - return current;
    - do it in an atomic manner !: `1 싸이클에 끝남`
  - How to use it?
    - old_value = *address; // not CAS
    - new_value = … my_calc … with oldVal ;
    - back = atomicCAS( address, old_value, new_value );
    - `if` (old_value == back) success!
  - in "device_atomic_functions.h"
    - TYPE = int, unsigned
    - TYPE atomicAdd(TYPE* address, TYPE val);
    - TYPE atomicSub(TYPE* address, TYPE val);
    - TYPE atomicExch(TYPE* address, TYPE val);
    - TYPE atomicMin(TYPE* address, TYPE val);
    - TYPE atomicMax(TYPE* address, TYPE val);
    - unsigned atomicInc(unsigned* address, unsigned val);
    - unsigned atomicDec(unsigned* address, unsigned val);
    - TYPE atomicAnd(TYPE* address, TYPE val);
    - TYPE atomicOr(TYPE* address, TYPE val);
    - TYPE atomicXor(TYPE* address, TYPE val);
    - TYPE atomicCAS(TYPE* address, TYPE compare, TYPE val);
  - in "sm_60_atomic_functions.h" `60 아키텍처 부터는 다른 타입도 지원함` 
    - TYPE = int, unsigned, unsigned long long, float, double
    - 불가능한 조합도 있음
      - 예: atomicCAS 는 int 형만 가능
  - atomic op 를 사용하려면, 
    - "device_atomic_functions.h" 와 "sm_60_atomic_functions.h" 둘다 include 해야 함
    - 컴파일 하려면, CUDA 아키텍쳐가 `sm_60` 이상 이어야 함 
      - `nvcc -w -arch sm_89 -o count_block count_block.cu`: `-w -arch sm_89` 추가 하거나.
      - CMakeLists.txt 에. `set(CMAKE_CUDA_ARCHITECTURES 89)  # This is for CUDA 11.2 and newer` 추가 

***Type Conversion Instrinsics 타입 변환***
- device function 내에서의 type `reinterpretation` intrinsic functions 내장 함수
  - `as` 가 이름에 붙는 함수
  - A as B: A 타입의 숫자의 비트패턴을.. B 타입의 숫자로 바꿔서.. 처리함
    - 비트패턴이 유지되므로. 정밀도가 유지 됨
    - ex) float (0.15625): 비트패턴 0x3E200000 => int (1,042,284,544): 비트패턴 0x3E200000 => float (0.15625): 비트패턴 0x3E200000
      - 복구 가능 
  - __device__ int __float_as_int( float x );
    - reinterpret bits in a float as a signed integer.
  - __device__ float __int_as_float( int x );
    - reinterpret bits in an integer as a float.
- type `conversion` intrinsic functions
  - rounding mode 에 따라 int to float conversion
  - ex) float (0.15625) => int conversion: 0 => float (0.0)
    - 복구 불가능 
  - __device__ float __int2float_rd( int x );
  - __device__ float __int2float_rn( int x );
  - __device__ float __int2float_ru( int x );
  - __device__ float __int2float_rz( int x );


***CUDA Atomic Functions Again***
- original atomic functions – device-wide atomic
  - global variable 을 대상으로 하므로.
  - device 내의 모든 thread 들에 대해서 atomic !
  - 그래서, `기존의 atomic 함수`는 `global 변수 용으로 최적화` 됨
- - block-wide atomic functions: `SM 6.0 이후`
- shared memory variable 에 대해서는 더 효율적임
- thread-block 내의 thread 들에 대해서만 atomic 을 보장
- `TYPE atomicAdd_block( TYPE* address, TYPE val );`
  - `쓰래드 블락 내에서, 즉, 쉐어드 메모리 내에서 사용하는 경우. 이 버전이 최적화`
- system-wide atomic functions: `SM 6.0 이후`
  - system 내의 모든 CPU, GPU 에 대해서, atomic
    - 즉, 여러개 CPU, 여러개 GPU 일때도 atomic 사용 가능해야 하니.
    - 그래픽 카드끼리 서로 통신해서, atomic 함수 하나만 실행시켜주는 식으로 발전 함
    - 하나가 아니라. 여러개의 그래픽 카드가 협력하니, 이 연산은 더 느려지겠지만, 대신 필요할 때가 있을것임.
  - `TYPE atomicAdd_system( TYPE* address, TYPE val );`



***example: number counting***
  - non-atomic version: `count_race.cu`
    - 실행 시간: 95,040 usec
    - NUM 개수: 67,108,864
    - num thread launched = 67,108,864 
    - cout 결과: 995 
      - => 64million 즉 67,108,864 가 아님, `count 실패`  
  - atomic version: `count_atomic.cu`
    - 실행 시간: 93,213 usec
    - NUM 개수: 67,108,864
    - num thread launched = 67,108,864
    - cout 결과: 67,108,864 => `count 성공`
  - shared-memory atomic version: `count_shared.cu`
    - 실행 시간: 76,880 usec
    - NUM 개수: 67,108,864
    - num thread launched = 67,108,864
    - cout 결과: 67,108,864 => `count 성공`, `속도도 개선` 
      - 그런데, 쉬운 문제일 경우, syncthread() 때문에 느려지는 경우도 있지만,...
      - 복잡한 연산이 많은 로직의 경우, shared memory 사용하면 빨라진다.
  - block-wise atomic version: `count_block.cu`
    - 실행 시간: 72,625 usec
    - NUM 개수: 67,108,864
    - num thread launched = 67,108,864
    - cout 결과: 67,108,864 => `count 성공`
      - => `original shared mem 보다 약간 빠름`
      - => `누적되면 훨씬 빨라짐`.
  - CAS-based re-implementation: `count_cas.cu`
    - CAS 연산, atomicAdd를 직접 작성 가능
    - NUM 개수: 67,108,864
    - num thread launched = 67,108,864    
    - cout 결과(구현 함수): 67,108,864.0000 => `제대로 값이 나온다`
      - 실행 시간: 299,428 usec
        - `확실히 CUDA에서 제공하는 함수가 더 빠르다.!!` 
        - 왜? 어셈플리로 구현되어 있으니.
  

  

[Return Par5 Atomic Operation](../README.md)  