# Reduction Problem

***Reduction Problem***
- We need to summarize a set of input values.  
  - 여러 숫자들을 종합해서, 1개의 지표로 제시 (summarize 1)
- a set of input values → a single value
- 예를 들면
  - sum = a0 + a1 + ... + a(n-1)
  - prod = b0 x b1 x ... x b(n-1)
- 통계 statistics 분야에서도 자주 발생
  - max 최대, min 최소, average 평균, standard deviation 표준편차, 



***Sequential Reduction Algorithms***
- 전통적인 계산 방법
  - step 1: initialize – 계산 결과를 `초기화`
    - 예: sum = 0, product = 1, max = –∞, min = +∞, . . .
  - step 2: `반복`해서 iterate 결과를 업데이트
    - 예: sum new = sum old + value
- sequential implementation:
  - a single for-loop → `O(n) operations`
  - ```c++
    sum = 0;
    for (i = 0; i < n; ++i) 
    {
        sum = sum + value[i];
    }
    ```

***Sport Tournaments***
- sequential reduction 의 개선책?
  - 좀더 짧은 시간 / 작은 비용으로 우승자를 뽑자 → 토너먼트
  - 참고: 다른 방법은 .... 리그.., 
  - 즉.. tree 구조..

***Parallel Reduction Algorithms***
- 병렬로 처리하는 토너먼트 알고리즘
- 그려놓고 보면, 트리 형태 임 (full binary tree) → `트리 알고리즘` 이라고도 함
- `O(log n) steps`
- a quick analysis
  - number of operations: O(n) `실제로 필요한 연산의 개수`
    - (1/2) n + (1/4) n + (1/8) n + … + (1/n) n = (1 – (1/n)) n = (n – 1)
  - number of steps: log_2(n) `대신 병렬처리가 가능하니, 스텝수는 줄일 수 있다.`
    - 이론상, 1 million data = 2^20 data → `20 steps` 
    - (`단, 무제한 core 필요` <- `물리적인 제한`)
- 비교: `sequential` algorithm: O(n) operations, in `O(n) steps`
- This is a `work-efficient` parallel algorithm
  - 즉, sequential version 에 비해 추가 작업없이, 효과적으로 처리
  - `연산량이 안 늘리고도, 더 빨라지는` 아주 좋은 알고리즘이라고 해서 `work-efficient` 한 `패러럴 알고리즘` 이라고도 함 
  - cf) sort 등은 연산량 늘리면서 (더 많은 연산을 해야하는) parallel 사용해서, 빠르게 하는 algorithm 도 있음 

***Example (a sum reduction) : total sum of 16M elements (1)***
- CPU version `sum_host.cu`
  - use getSum(...) in "./common.cpp"
  - 실행 시간: 7,752 usec
- atomic op `sum_atomic.cu`
  - 16 million 개의 쓰래드가 돌아갈거고, atomicAdd 사용 => 꽤 느려질꺼임.
  - 구현 단계
    - device global variable
      - __device__ float dev_sum = 0.0f;
    - add all elements in parallel
      - atomicAdd( &dev_sum, pData[i] );
    - 16M atomic addition …
  - 실행 시간: 25,922 usec 
- shared atomic op `sum_shared.cu`
  - 구현 단계
    - device global variable
      - __device__ float dev_sum = 0.0f;
    - shared memory partial sum
      - __shared__ float s_sum;
    - add all elements in a thread block: `1,024 additions per block` 
      - `16k 개의 쓰레드 블럭`이 있을것임 
      - atomicAdd_block( &s_sum, pData[i] );
    - then, `16K` atomic addition, again
      - atomicAdd( &dev_sum, s_sum );
    - 실행 시간: 2,810 usec
- reduction `sum_reduce.cu`
  - parallel implementation 실제 구현:
    - 각 thread 가 2개의 숫자를 더한다.
    - thread 숫자를 절반으로 줄인다. → `이제 thread 의 절반은 불필요!!`
    - log(n) 번 반복
  - Assume an `in-place reduction` using `shared memory`: `쉐어드 메모리 내에서만 이 토너먼트를 돌리게 해야 함`
    - The original vector is in device `global memory`
      - 즉, 글로벌 메모리는 이전 과 같이, atomicAdd로 사용 
    - The shared memory is used to hold `a partial sum vector`
      - 1024개만 일단 쉐어드 메모리에서 토너먼트로 돌려보자는게 아이디어.
    - Each step brings the partial sum vector closer to the sum
    - The final sum will be in `element 0`
      - `최종 결과`는 제일 `첫번째 element` 에 결과가 저장되는 식으로 구현.
    - Reduces global memory traffic due to partial sum values
  - 스트라이드 사용 방식 
    - step1: stride 1
    - step2: stride 2
    - step3: stride 4
    - step4: stride 8
    - step5: stride 16
    - ...
  - 다시 말하면, stride 보폭 값의 변화는
    - stride = 1 → for threads with (tx % 2 == 0)
    - stride = 2 → for threads with (tx % 4 == 0)
    - stride = 4 → for threads with (tx % 8 == 0)
    - stride = 8 → for threads with (tx % 16 == 0)
    - generally, → for threads with `(tx % (2 * stride) == 0)`
  - 실제 구현: `interleaved addressing`
    - ```c++
      for (register unsigned stride = 1; stride < blockDim.x; stride *= 2) 
      {
        if (tx % (2 * stride) == 0) 
        {
            s_data[tx] += s_data[tx + stride];
         }
        __syncthreads();
      }
      ```
  - 구현 단계
    - device global variable
      - __device__ float dev_sum = 0.0f;
    - add all elements in a thread block: tournament addition
      - no atomic addition !
      - instead, we need `__syncthreads()` <= `꼭 필요`
    - then, 16K atomic addition, again
      - atomicAdd( &dev_sum, s_sum );
  - 실행 시간: 512 usec (= 0.5 ms) 
    - `압도적으로 빨라짐!!`

***Example (a sum reduction) : total sum of 16M elements (2) - reduction 로직도 더 빠르게.)***
- reversed reduction
  - 패러럴 구현
    - 각 쓰래드가 2개의 숫자를 더한다
    - 쓰래드 숫자를 절반으로 줄인다 -> 이제 쓰래드의 절반은 불필요
    - log(n)번 반복
  - keep the active threads consecutive 연속적으로
    - 불필요한 쓰래드를 모아보자
    - warp 내의 모든 쓰래드가 불필요해지면, early terminate 가능!
    - 해결책: stride를 반대로 계산
  - stride 값의 변화
    - stride = 8 for threads with (tx < 8)
    - stride = 4 for threads with (tx < 4)
    - stride = 2 for threads with (tx < 2)
    - stride = 1 for threads with (tx < 1)
    - generally, for threads with (tx < stride)
  - 실제 구현
    - ```c++
      // >>= 1: 나누기 2 의 의미
      for (register unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1)   
      {
          if (tx < stride) 
          {
              s_data[tx] += s_data[tx + stride];
          }
          __syncthreads();
      }
      ```
  - 실행 시간: 408 usec
- add first: 최초에 더하기부터 하는 방식
  - reversed reduction 의 비효율적인 부분 개선
    - ```c++
      // each thread loads one element from global to shared memory
      s_data[tx] = pData[i]; // <--
      __syncthreads();
      // do reduction in the shared memory
      for (register unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) 
      {
          if (tx < stride) 
          {
              s_data[tx] += s_data[tx + stride]; // <--
          }
        __syncthreads();
      }
      ```
      - 모든 쓰래드가 글로벌 매모리 -> 쉐어드 메모리 복사에 참여
      - 쓰레드의 절반은 for-loop 의 첫번째 수행시에 이미 불필요
    - 개선책: 
      - 쓰래드가 2번 load 하고, 이것들을 일단 더한다(add first)
      - 쓰래드 개수가 절반으로 줄임
  - 실행 시간: 283 usec `reversed reduction 보다도 빨라짐` 
- last warp
  - for a 1,024 thread block,
    - 1024, 512, 256, 128, 64, 32 consecutive threads are active in each step
    - less than or equal to 32 threads → a warp 
      - `쓰래드가 32개 또는 그 이하 남았을경우, 1개의 와프로 돌아가므로, 이때는 __syncthreads() 안해도 된다.!`  
  - for a warp,
    - instructions are synchronous within a warp
    - We don't need to __syncthreads()
    - We don't need "if (tx < s)", because it doesn't save any work
      - `하나의 와프 warp 내에서라면,` 
      - `어차피 최종 결과인 하나의 레지스터만 사용해서 계산하고 있고`
      - `32개의 쓰래드가 동시에 수행시간을 쓰고 있으니깐,` 
      - `if 문 체크를 안해도 문제가 없을것이다!`
  - 해결책: Let’s unroll the last 6 iterations of the inner loop
    - `warpSum()` 함수 구현 확인! 
  - 실행 시간: 208 usec
- warp shuffle
  - warp 내에서 32개 쓰레드가 서로 더하기 하는 부분은 필요하다면, `__shfl_xor_sync` 함수 사용하면, 쪼금 더 빨라진다.
  - 일반적으로는 사용하지 않는 고급 스킬 이라서, 몰라도 됨!
  - 실행 시간: 427 usec
    - 더 빨라진다고 하는데, 나의 경우는 더 느림.
- two step: 2단 커널 방식
  - 실행 시간: 200 usec
    - 마지막에 atomicAdd를 쓰지않고, 커널 레벨에서 2레벨로 sum을 하도록 해서,
    - 즉, 작은 커널이 2레벨로 reduction 을 하는 것을 완전히 구현해서,
    - atomicAdd() 없이 돌아가도록 하면서, 조금 더 빨라졌음


***volatile keyword***
- example: volatile int k;
  - meaning: variable k can be updated by an external device/process/thread.
    - 이 변수는 다른 디바이스나, 프로세스나, 쓰래드에서 업데이트 할수도 있다고 컴파일러에 알려줌
    - 그래서, 이 변수는.. 컴파일러가 최적화 하지 않음.
  - effect: no optimization by the compiler
  - can be used in C / C++ / Java / CUDA
- 예: memory-mapped I/O 상황
  - C/C++ 에서, 외부 장치로 data 전송
    - ```c++
       volatile unsigned* addr = 0x308;
       *addr = 0x48; // H
       0x308 network device
       *addr = 0x69; // i
       *addr = 0x04; // end of transmission: `EOT` 
      ```
  - compiler's optimization:
    - ```c++
      *addr = 0x04; // end of transmission 
      ```
      - 컴파일러 최적화에 의해.. Hi 가 삭제.. `EOT` 만 전송됨 
  - C++/CUDA example code:
    - a[tx] += a[tx + 2]; // step 1
    - a[tx] += a[tx + 1]; // step 2
  - C++/CUDA compiler can optimize it as: `컴파일러의 오버 옵티마이즈 되면`
    - a[tx] = a[tx] + a[tx + 1] + a[tx + 2];
    - failure if "a[tx + 1]" is updated at the time of step 2
  - C++/CUDA solution:
    - volatile int a[64];
    - no optimization any more !

***Warp Shuffle Functions***
- special warp shuffle functions
  - exchange a (register) variable between threads within a warp
- T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
  - direct copy from indexed lane
- T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
  - copy from a lane with lower ID relative to caller
- T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
  - copy from a lane with higher ID relative to caller
- T `__shfl_xor_sync`(unsigned mask, T var, int laneMask, int width=warpSize);
  - copy from a lane based on bitwise XOR of own lane ID
  - warp 내에서 32개 쓰레드가 서로 더하기 하는 부분은 필요하다면, `__shfl_xor_sync` 함수 사용하면, 쪼금 더 빨라진다.
  - 일반적으로는 사용하지 않는 고급 스킬 이라서, 몰라도 됨!

***Kernel Decomposition***
- two-level kernel launch
  - kernelSum( ): gets the sum of 1,024 elements
  - two-level invocation → (1,024) 2 = 1M elements

[Return Par5 Atomic Operation](../README.md)   