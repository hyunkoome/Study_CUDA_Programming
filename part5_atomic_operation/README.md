# Atomic Operation

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

  
[Section33: GEMV(general matrix to vector multiplication)](./doc/section27.md)

  
[Return Main Readme](../README.md)  


