# Memory Structure

***Matrix Transpose 전치 행렬***
- C = A^T = A^tr
  - C[i,j] = A[j,i]
  - for simplicity, we assume `square matrices`
  - A[y][x] = A[y * WIDTH + x]
    - => C[y][x] = A[x][y] = A[x * WIDTH + y]


***matrix transpose problem 전치 행렬 문제***
- 16k x 16k
- CPU version
  - `./실행파일 16k` 로 실행 
  - 실행시간: 3,287,833 usec = 3.28초
- CUDA naive version – global memory
  - 두가지 matrix 모두 pitched 된거 고려해서 구현 필요! (글로벌 index 사용)
  - 실행시간: 10,257 usec = 0.010초
- CUDA shared memory, naive version
  - tiled approach 사용 (글로벌 index, local index 사용)
  - 실행시간: 7,602 usec = 0.0076초
- CUDA shared memory, optimized version
  - 실행시간: 3,236 usec = 0.003236초
  - (행단위로 ) 가로로 저장을 해야, 속도가 빠르므로, 아래와 같은 트릭을 사용  
    - 읽어오는 부분은 M[y][x] = A[y][x]
    - 저장하는 부분은 C[y][x] = M[x][y]
- CUDA shared memory, `bank conflict` resolved version 메모리 뱅크 출동 현상
  - 쉐어드 메모리의 특성을 잘 이용하는 방법: `back conflict 상황은 피하게!!`
    - 세어드 메모리는 물리적으로 32개의 bank로 구성 (SP가 32개 이므로.)
    - 4개씩 끊어서, 4 bytes 씩 끊어서, 이 back 에 쫙~ 넣어놨음
    - bank 단위로 독립적으로 메모리를 access 할 수 있음
    - 그래서, back confict 가 없는 경우는. 가장 빠름. 
      - 32개 데이터를 32개 뱅크에서 1개씩 나눠서 갖고있어서 
        - 32개 SP에서 1개씩 엑세스해서 처리하니, 1싸이클에 끝남
    - 그러나, back confict 가 있는 최악 의 경우는. 가장 느림.
      - 32개 데이터를 1개 뱅크에서 모두 갖고 있어서.
      - 1개 SP에서 32번 처리해야 함. 32 싸이클을 거쳐야 끝남 
    - 2-way bank conflicts: 2배 느려짐
    - 16-way bank conflicts: 16배 느려짐
  - Bank Conflict: 2D Case
  ```c++
  __shared__ float mat[32][32];
  mat[threadIdx.y][threadIdx.x] = . . . ; // no bank conflict
  . . . = mat[threadIdx.x][threadIdx.y]; // bank conflict
  ```
   - threadIdx.x = 0, . . ., 31, for a specific threadIdx.y,
     - `mat[0][ty], mat[1][ty], . . . , mat[31][ty]`: 모두 1개의 bank 에 몰림
     - => 해결책: `__shared__ float mat[32][32+1];` 으로 선언 
       - `mat[0][ty], mat[1][ty], . . . , mat[31][ty]`: 완벽히 분산 됨!
  - 실행시간:
    - `__shared__ float mat[32][32];`: 3,224 usec = 0.003224초
    - `__shared__ float mat[32][32+1];`: 2,787 usec= 0.002787초



[Return Par4 Matrix Multiply](../README.md)  