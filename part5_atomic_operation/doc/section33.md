# GEMV: general matrix to vector multiplication

***GEMV operation***
- generalized matrix-vector multiplication `메트릭스와 벡터 사이의 곱하기`
- z ← α A x + β y
  - A : matrix
  - x, y, z: vector
  - α, β : constant
- BLAS level 2 operation
  - sgemv (single precision)
  - dgemv (double precision)
  - cgemv (complex)

***test for 행렬 (16K-by-16K) x 벡터 (16K) case***
- host version `gemv_host.cu`
  - 실행 시간: 218,394 usec
- CUDA global memory `gemv_cuda_globalmem.cu`
  - grid 구조
    - z 베터 기준으로 실행
    - y 축 방향으로 launch
  - 실행 시간: 7,481 usec
- CUDA transpose matrix `gemv_cuda_transpose.cu`
  - memory access 에 대한 고려
    - matrix A를 그대로 사용하는 경우
      - warp 내의 인접한 쓰래드 들은 A 의 아래쪽 자료를 요구
    - matrix A의 transpose 사용
      - memory coalescing 달성!
  - 실행 시간: 1,621 usec
    - 여기서는 transpose 연산은 미리 되어있다고 보고, 실행 시간을 계산 함(미포함)
    - transpose 연산 시간: 2,679 usec
- CUDA tiled `gemv_cuda_tiled.cu`
  - 쓰래드 블럭에서 반복해서 사용하는 부분이 vector x 임
    - 이 부분을 쉐어드 메모리 s_X[블럭 사이즈]로 바꾸고
      - s_X 를 읽어올때, 블럭 사이즈 만큼 읽고, for loop 반복
  - tiled approach
    - tile
      - number of tiles: ntiles = ⎾matrix_size / tile_size ⏋
    - tile 내의 element 수
      - 보통의 경우: tile_size
      - 마지막 tile: last_elem = matrix_size – (ntiles – 1) * tile_size
        - 1 ≤ last_elem ≤ tile_size
  - 실행 시간: 1,242 usec
- partial sum 으로 계산하는 시도 `gemv_cuda_2d_kernel_partial.cu`
  - 우리가 지금 쓰래드를 launch 할때, 이러지 말고, 어차피 지금, tiled approach를 하고 있으니
  - 처리하는 각 tile을 그냥 할꺼번에 돌려보자
  - 여러개의 쓰래드로 분리해서 계산
    - 쓰래드가 각각 s_X 사용
    - fpr loop 돌리지 말고, 각 tile 별로 처리하는 걸, 
    - x 디멘션이 남아있으니, 여기를 2d 커널 형태로 돌려버리고
    - 각 x dimension 돌아가면서, 
    - 계산한 결과는 atomicAdd()로 더해려서
    - 총합을 구해보자
  - 실행 시간: 6,345 usec 
    - memory coalescing 문제로 의외로 느리다.
    - 즉, `memory coalescing 을 고려하면,` 
      - `너무 커널을 세분화 해서 돌리는 것 보다는,`
      - `차라리, tile을 쓰는 정도로 접근하는게 낫다.`


[Return Par5 Atomic Operation](../README.md)   