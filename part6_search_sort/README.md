# Search & Sort

[Section28: Control Flow](./doc/section22.md)
- if 문 과 for loop 문 어떻게 최적화 할것인지?    
- 숫자를 서로 섞어야 하는 예제(shuffling cases)에서..even-odd 와 half-by-half 어떻게 구현하는지, 그리고 수행 시간..
- shared 메모리를 사용하는 경우라면, `half-by-half`를 사용하는 `even-odd` 보다 조금더 빠르다.!!

[Section29: Atomic Operation](./doc/section23.md)
- race conditions
  - 병렬처리에서 여러 threads 에서 하나의 변수를 서로 읽거나, 업데이트 하고 싶어할때 발생
      - shared 메모리, global 메모리 모두 대상 임
  - global communication
      - 서로 다른 스래드 블록 간의 통신
      - 글로벌 메모리를 대상으로..
- 이 해결잭으로: atomic operation을 사용함
  - memory locking
    - 전통적인 해결책
    - lock - update - unlock
    - lock 실패시, 기다림
  - `atomic operation`
    - 새로운 해결책 => CUDA 도 채택
    - atomic update (a single operation)
        - 어셈블리 명령어로.. 1싸이클에 완료됨.
    - 동시 시도 가능. 1개만 성공
- `결론적`으로, 
  - `여러 쓰래드에서 하나의 변수에 접근해서 값을 읽거나 업데이트 할때`  
  - `쓰래드 블럭` 사용하고, 즉, `쉐어드 메모리`에서 
  - `atomic 연산 사용`하면, `엄청 빨라짐`.

[Section30: Histogram Problem](./doc/section24.md)
- atomic operation 사용하여 histogram 구하기
  - 4바이트 처리하면. 빨라지고
  - `shared 메모리에서, atomic 연산하면서, 4바이트 처리하면 가장 빠르다!!`

[Section31-32: Reduction Problem](./doc/section25.md)
- Reduction Problem
  - Sequential Reduction Algorithms
  - Parallel Reduction Algorithms
- Example: total sum
  - CPU version
  - atomic op
  - shared atomic op
  - reduction: `이것만 해도. 엄청 획기적으로 빨라진다!`
  - reversed reduction `reduction 적용순서를 반대로 하거나`
  - add first `혹은, 미리 두개를 더하는 add first 전략을 쓰거나`
  - last warp `혹은, last warp를 빨리 처리하는 기법을 도입하거나`
  - warp shuffle `warp를 셔플 시키는 것도 있었고`
  - two step `two step 어프로치로, 작은 커널에서 reduction program을 two level로 돌리는 것을 적용`
  
[Section33: GEMV(general matrix to vector multiplication)](./doc/section27.md)
- GEMV operation
  - generalized matrix-vector multiplication
- test for 행렬 (16K-by-16K) x 벡터 (16K) case
  - host version
  - CUDA global memory
  - CUDA transpose matrix
    - `transpose 테크닉을 사용해서.`
    - `데이터를 가져올때 메모리를 가로로 읽게 하면,`
    - `속도가 엄청 개선됨.`
  - CUDA tiled
  - CUDA 2D kernel
    - `memory coalescing 을 고려하면,`
    - `너무 커널을 세분화 해서 돌리는 것 보다는,`
    - `차라리, tile을 쓰는 정도로 접근하는게 낫다.`
  
[Return Main Readme](../README.md)  


