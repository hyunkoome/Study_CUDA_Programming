# Memory Structure

***Matrix Multiplication***
- for simplicity, we assume `square matrices` `정방 행렬`
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
   
***3중 for loop의 개선책*** 검색 키워드: `loop (nest) optimization`
- in your C/C++ text book,
  ```c++
  for (register unsigned y = 0; y < matsize; ++y) {
      for (register unsigned x = 0; x < matsize; ++x) {
          for (register unsigned k = 0; k < matsize; ++k) {
  ```
- L1/L2 cache 의 효율적 시용
  - k 를 loop 제일 바깥으로!
  - B 행렬이 cache 에 오래 머물도록
  ```c++
  // C 전체를 0으로 clear 해놓고 시작함 
  for (register unsigned k = 0; k < matsize; ++k) {
    for (register unsigned y = 0; y < matsize; ++y) {
      for (register unsigned x = 0; x < matsize; ++x) {          
  ``` 
  - 즉, C 행렬에 어떤 값을 구할때. 한꺼번에 이 값을 구하는 게 아니고,
    - 일부 구하고 차곡차곡 더해 나갈 것임
    - C 의 1개 원소를 구하는 것이 아니라, C의 한 row(줄) 전체를 계산해보자!
  


***Matrix Multiplication***
- matrix size: 4k(=4096) x 4k(=4096)
  - `./실행파일 4k`
- CPU implementation
  - native triple for loop `3중 for-loop 문 사용`
    - 실행 시간: 390,401,410 usec = 390초 = 6분 30초  
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
    - `./matmul_cuda_globalmem.cu 8k`
      - 실행 시간: 333,206 usec = 0.333206초
  - shared memory - tiled approach
    - warp 단위로 처리 중
      - `읽어올때`, `계산할때`, 각각 `다른 트릭`을 사용함!
        - 1) tile 을 갖올때는 row-major (옆줄 단위로) 로 가져오는 게 빠름!
        - 2) 싱크 쓰래드 사용해서 동기화!
        - 3) 계산 processing 은 각각 row, column 으로
    - `./matmul_cuda_more_improve_alignedTile.cu 8k` 
      - tile 크기를 32 x 32 로 assume
      - matrix 크기를 32의 배수로 assume (4k x 4k 사용)
      - 실행 시간: 267,511 usec = 0.267511 
    - `./matmul_cuda_more_improve_aligned2.cu 8k 16`
      - tile 크기를 변화 시킬수 있게 수정 (16x16 tile 도 가능)
      - 그러나, `TILE_WIDTH 가 32보다 클수는 없음!!`, 최대 32임 
      - 실행 시간: 269662 usec = 0.269662
        - tile size 줄인다고, 속도개선이 되지는 않더라!
      - `./matmul_cuda_more_improve_aligned2.cu 8k 8`
        - 실행 시간: 555905 usec = 0.555905
        - `tile size 줄인다고, 속도개선이 되지는 않더라!`
        - => `최신 gpu 에서는 tile size를 크게 잡는게, 속도가 빨라지더라~!`
    - `./matmul_cuda_sharedmem_tiled_approach.cu 8100`
      - matrix 크기를 임의로 변경 가능 (4000 x 4000 도 가능)
      - 실행 시간: 271878 usec = 0.271878
        - 약간 느려지는 정도로 실행 됨, 대신 임의의 크기도, 계산 가능!!
        - 속도 저하는 어쩔 수 없음, if 문으로 계속 체크하는데, 속도가 비슷하게 나올수 없음 
    - `./matmul_cuda_sharedmem_tiled_approach2.cu 8100 20`
      - tile 크기 임의 설정 가능
      - matrix 크기 임의 설정 가능
      - 실행 시간: 269707 usec = 0.269707 
        - 속도저하는 어쩔수 없지만, 행렬 크기나, block 크기 모두 변화 가능하다. 
        - 그래도, cpu 또는 global 메모리 보다는 빠르다!
  - more improvements



[Return Par4 Matrix Multiply](../README.md)  