# Matrix Multiply

***Matrix Copy***
- simply copy a matrix to another
  - C[i,j] = A[i,j]
  - `pitched matrices` for the best performance
  - for simplicity, we assume `square matrices`
- 다른 matrix 연산의 입장에서는?
  - theoretical limit 이론적 한계for matrix operations
  - the best score 최고 기록 for any matrix operations 행렬 대입 연산이 최고 기록이다.
  - 따라서 다른 연산, 즉 더하기 연산 등은 이 행렬 대입연산의 속도에 근접할 수록 최적화가 잘 된것이고
  - 이 기록을 추월했다면, 만세를 부를게 아니라, 뭔가 잘 못된 것임
    - 즉, 연산의 물체를 발견할 수 있는 어떤 기준이 될수도 있음


***matrix copy – theoretical limit***
- CPU version
  - 실행 시간: 0.43초
- memcpy version
  - 실행 시간: 0.45초
  - CPU에서 이중 루프 돌린거 보다 메모리 카피 쓴게 의외로 약간 느림
    - 이 결과는 메모리 카피가 운영체제에서 지원 되면서, 굉장히 효과적으로 짜여있긴하지만, 
    - 경우에 따라서는 cashe 상황이나, 혹은 컴파일러가 for 루프를 얼마나 optimize 했느냐에 따라, 
    - 이런 관계 때문에, `경우에 따라서는 우리가 for 루프 돌려서 직접 짠 것이 더 빠를수도 있다는 것을 보여줌`
- CUDA naive version – `global memory`
  - naive : (전문적) 지식이 없는
  - pitched matrix 를 사용하는 것이 속도가 빠르므로.. 이 방법을 사용
    - cudaMallocPitch(), cudaMalloc3D() -> 바이트 단위
    - data: float -> sizeof(float) = 4 bypes
  - A[y][x] 의 위치 계산: `byte 기준`
    - offset = y * dev_pitch + x * sizeof(float); // in byte
    - *((float*)((char*)A + offset) = 0.0f;
  - A[y][x] 의 위치 계산: `index 기준`
    - assert(dev_pitch % sizeof(float) == 0);
    - pitch_in_elem = dev_pitch / sizeof(float);
    - idx = y * pitch_in_elem + x;
    - A[idx] = 0.0f;
  - 실행 시간: 0.0067초
    - 분명히, cpu 버전들 보다 빨라짐
    - `글로벌 메모리`를 이용해서, `글로벌 메모리로 읽은` 다음에, `글로벌 메모리로 복사`하는 예제임
- CUDA `shared memory` version <-- 이 것이.. `속도 측정에 기준이 되는 레퍼런스가 됨`
  - `tiled approach` 사용 해야함
    - Tile Size 잡을때,
      - 가능하면, for simplicity, we assume square matrices, 정방 행렬로 잡아라!!
      - tile : 정사각형 모양은 선호
      - `__shared__ float s_mat[32][32];`
        - 32 x 32 = 1,024 = `최대 쓰래드의 개수`
        - 1,024 = maximum number of threads in a thread block
        - 32 = a single warp
      - old code 에서는 16x16 또는 32x16 사용
        - 이유: maximum number of threads in a thread block `was 512`
        - 16 x 16 = 256
        - 32 x 16 = 512 → 1개의 thread 에서 2개씩 처리 � → 32x32 로 작동
  - `2d matrix in the global memory`
    - 글로벌하게 전체 그리드 상에서 하나의 block을 잡을 거고
    - 이때, global index 사용
      - gx = blockIdx.x * blockDim.x + threadIdx.x;
      - gy = blockIdx.y * blockDim.y + threadIdx.y;
  - `2d sub-matrix in the shared memory`
    - 그 block 이 shared 메모리로 카피되서 올겁니다.
    - 이때, 자기 쓰래드 인덱스를 이용해서(local index 사용),
      - = 어느 위체에 shared 메모리 어느 위치로 가져와야 될지는 쉐어드 메모리 주소 사용하고
        - tx = threadIdx.x;
        - ty = threadIdx.y;
    - 어느 위치에서 가져올 지는 글로벌 메모리 인덱스 사용
      - gx = blockIdx.x * blockDim.x + threadIdx.x;
      - gy = blockIdx.y * blockDim.y + threadIdx.y;    
  - `another 2d matrix in the global memory`
    - 카피가 끝나면, 역으로 global index 사용해서.. 
      - gx = blockIdx.x * blockDim.x + threadIdx.x;
      - gy = blockIdx.y * blockDim.y + threadIdx.y;
    - 다시, 쉐어드 메모리 데이터를 타겟이되는 글로벌 메모리로 카피
  - 실행 시간: 0.0073초
    - 글로벌 메모리 사용 버전과 비교하면, 다소 느리지만, 
    - 그 이유는 shared 메모리를 한번 더 거쳤으니까, 느릴수 밖에 없다.
    - `그러나, 일반적으로 shared 메모리에서 행렬 copy가 아닌, 실제 연산등이 추가되므로.`
    - `이 방법을 사용하는 것을 연습하기를 권장 함`
- CUDA memcpy2D 쿠다 커널 직접 카피 (`cudaMemcpy2D()`)
  - 실행 시간: 0.0056초
  - CUDA에서 글로벌 메모리 쓰는 버전이나, 쉐어드 메모리 쓰는 버전보다. 빨라짐
  - 그래서, `사실은 이것이.. 우리가 CUDA 프로그래밍할때 기준이 되는, 속도 레퍼런스가 됨`
    - 이것 보다 더 빠른 알고리즘을 우리가 CUDA에 구현하는 것은 불가능함.





[Return Par4 Matrix Multiply](../README.md)  