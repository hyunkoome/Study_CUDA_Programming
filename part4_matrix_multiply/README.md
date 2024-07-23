# Matrix Multiply    

[Section 16: matrix addition: 행렬끼리 더하기](./doc/section16.md)
- `CUDA로 프로그래밍 할때는, 항상 array index 주의해라!`
- `row 가 y 이고, column 이 x 라는 것을 주의 해서`
- `y * width + x 로 계산 해야 한다!`

[Section 17: cuda malloc 2D: CUDA 전용의 2D 메모리 할당 함수](./doc/section17.md)
- Memory Coalescing 코얼리싱, 합병 
- 2D 행렬에서 pitched point 사용법 

[Section 18: 3D filter 적용](./doc/section18.md)
- 3D 행렬 사용 및 pitched point 사용법
- 3D Pitched Matrix 함수들

[Section 19: memory_hierarchy 메모리 계층 구조](./doc/section19.md)
- Shared Memory in CUDA
- CUDA variable 변수

[Section 20~21: adjacent differences 인접 원소끼리 차이 구하기](./doc/section20_21.md)
- Data Access Patterns
  - 의외로 global 메모리를 사용하는 쪽이 더 빠름
  - 최신 CUDA 디바이스들은 L1/L2 cache 가 추가되고, 성능이 매우 좋음
  - device 마다 test 필요
  - 그러나, `복잡한 경우에는 shared memory를 쓰면 확실히 빠름`
- Race Condition: shared memory case
  - 해결 방법: Barrier Synchronization
    - __syncthreads() 사용해라.! 
    - 주의: `heavy operation 이어서, 필요할때만 쓸 것!!`
    - 전체적으로 모든 쓰래드가 조금씩 늦어짐!!
- Device Query 함수를 사용해서 CUDA 디바이스의 properties 를 얻자
- Pointers 가 shared 메모리등 , 특정 메모리 공간을 직접 가리키지는 않음
  - 포인터의 포인터. 즉 링크드 리스트나, 트리같은 구조는
    - 가능하면 GPU에서 쓰지 말라는게 가이드임
  - 모든 포인터가 글로벌 메모리를 가리키도록 하거나,
    - 모든 포인터가 쉐어드 메모리를 가리키도록 하는등, 심플하고 레귤러하게 쓰는 식으로 처리 해라!!
- kernel function call 시의 함수 파라미터 처리
  - 기본적으로 call-by-value
  - struct를 넘기면, 그대로 copy 됨
  - pointer 와 array 값은 모두 CUDA global memory space 로 assume 가정
  - 그래서, 커널에 어떤 파라미터로 뭘 넘길때는, 항상 글로벌 메모리에 있는 것을 넘겨줘야 된다!!

[Return Main Readme](../README.md)  


