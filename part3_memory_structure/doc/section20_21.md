# Memory Structure

### adjacent differences 인접 원소끼리 차이 구하기 

***Data Access Patterns***
- Carefully partition data according to access patterns
  - R/W within `each thread`
    - → `registers` (fast)
  - R/W & `shared within block` 
    - → `__shared__ memory` (fast)
  - `Read-only` 
    - → `__constant__ memory` (fast)
  - Indexed R/W within each thread, 즉 `배열`이라면,  
    - → `local memory` (slow)
  - R/W inputs/results `상당히 크고, 일반적인 데이터`라면 
    - → 변수 선언 보다는, `cudaMalloc` 을 사용하는 편이 확장성이 더 좋음 
    - → `cudaMalloc'`ed `global memory` (slow)
- a kind of divide and conquer approach
  - data: divided into smaller ones in the shared memory

***Race Condition: shared memory case***
- 해결 방법: Barrier Synchronization
  - CUDA intrinsic functions 내장 함수: 
    - 컴퍼일러가 해당 위치에 직접 asm 으로 코드를 삽입하는 형태.!
  - __syncthreads(): thread-block 내 모든 threads를 동기 맞추는 기능
    - 주의: `heavy operation 이어서, 필요할때만 쓸 것!!`
    - 전체적으로 모든 쓰래드가 조금씩 늦어짐!!

***실습***
- `adjdiff_host.cu`
  - 실행 속도: 26,259 usec = 0.026 초
- `adjdiff_cuda.cu`: global memory case
  - 실행 속도: 168 usec = 0.000168 초
- `adjdiff_shared_memory.cu`: shared memory case
  - 실행 속도: 192 usec = 0.000192 초
  - 더 느려 졌네. 
    - 위에서 언급한대로, `__syncthreads()` 하면서 늦어질 수도 있음 !
  - 그래도. shared memory 를 쓰자~~

  

[Return Par3 Memory Structure](../README.md)  