# Memory Structure

***GEMM***
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

***implementations (4k x 4k)***
- cpu: `gemm_cpu.cu`
  - 3중 for-loop
  - 실행 시간: 410,418,710 usec = 410초 = 6분 50초
- cpu outer-k `gemm_cpu_outerK.cu`
  - for-loop 순서 변경
  - 당연히, 순서 바꾸면 `10배 정도는 빨라져야 함!!`
  - 실행 시간: 91,769,364 = 91초 = 1분 31초 
- CUDA `gemm_cuda_globalmem.cu`
  - cuda global 메모리 사용 버전
  - grid 설계: 2D layout
    - matrix C 의 각 원소 (y,x) 마다 thread (block) 생성
    - 각 thread 는 for loop 로 row x column 계산
    - 3중 for loop 에서 바깥쪽 2중 for loof 는 `2D grid (block thread) 구조`로 대체
      - 안쪽의 하나의 for 루프만 커널 내에서 실행 하도록..
    - 이 모든 메모리 구조가 다 `pitched` 되어있다는 것은 신경써야 함!!
  - 실행 시간: 
    - (4k x 4k): 33,401 usec = 0.033 초
    - (8k x 8k): 330,170 usec = 0.330 초
- CUDA aligned tile `gemm_cuda_alignedTile.cu`
  - tile 크기를 32x32 로 가정
  - `matrix 크기 를 32의 배수로 가정`, (4k x 4k 사용)
  - 실행 시간
    - (4k x 4k): 36,619 usec = 0.036 초
    - (8k x 8k): 267,321 usec = 0.267 초
- CUDA general tile `gemm_cuda_tile.cu`
  - matrix 크기를 `임의로 변경 가능` (4000 x 4000 도 가능)
  - `gemm_cuda_alignedTile.cu` 보다 `다소 속도는 떨어지더라도, 임의로 변경가능하게 구현해야 할때는 대비하기 위해` 
  - 실행 시간
    - (4000 x 4000): 34,939 usec = 0.034 초
    - (4k x 4k): 39,068 usec = 0.039 초 
    - (8000 x 8000): 260,448 usec = 0.260 초
    - (8k x 8k): 298,069 usec = 0.298 초 
- CUDA general tile `gemm_cuda_tile_optim.cu`
  - matrix 크기를 `임의로 변경 가능` (4000 x 4000 도 가능)
  - `gemm_cuda_tile.cu` 좀더 `최적화`, 첫번째 if 문 삭제..
  - `gemm_cuda_alignedTile.cu` 보다 `다소 속도는 떨어지더라도, 임의로 변경가능하게 구현해야 할때는 대비하기 위해`
  - 실행 시간
    - (4000 x 4000): 34,356 usec = 0.034 초
    - (4k x 4k): 37,862 usec = 0.037 초
    - (8000 x 8000): 259,454 usec = 0.259 초
    - (8k x 8k): 282,715 usec = 0.282 초

[Return Par4 Matrix Multiply](../README.md)  