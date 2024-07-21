# Memory Structure

### cuda malloc 2D: CUDA 전용의 2D 메모리 할당 함수 
- Memory Coalescing 코얼리싱, 합볍
  - 메모리 효율을 높이는 방법
    - chunks 청크, 덩어리 사용
    - 보통 32/64/128 bytes 단위로 사용
  - `CUDA 에서는 각 warp은 128 bytes chunk transfer 로 사용`
    - 32 threads per warp
    - 4 bytes per float / int
    - so, 32 x 4 = 128 bytes read/write per warp
  - `CUDA 의 cudaMalloc() 함수`는 `256 bytes` boundaries `단위로 끊어져 있음`
    - CPU: word size 가 16/32/64bit
      - 2/4/`8 bytes` boundaries
    - intel SSE(Stream SIMD Extension) register size
      - 128 bit = 4 float = `16 bytes boundaries`
    - DirectX / OpenGL matrix functions
      - 4x4 matrix with float (4 bytes) = 4 x 4 x 4 = `64 bytes boundaries`
  - `cudaMallocPitch()` 등의 함수에서 실제 pitch 값을 return 해 준다.
    - 그때그때 시스템에 따라서, 다른 byte align 숫자를 쓰더라도, CUDA 가 졸려주는 pitch 숫자는
    - 현재의 CUDA execution 환경에서 최적화된, 그러니깐, CUDA 버전에 따라서, 시스템에 따라서
    - 최적화되어 있는 pitch 바이트 숫자를 돌려주니까 이 리턴 값을 사용하는게 제일 좋다.
    - `항상, cudaMallocPitch() 함수가 돌려주는 pitch 값을 사용하도록 하는게 가장 좋다.`
    - pitch 는
      - width x height 크기의 2D 배열을 잡을때, 
        - `width` 에 `낭비하는 영역 알파`를 `포함`한 전체 줄 `폭` 의 크기를 `pitch` 라고 함
        - `bytes 단위`
        - 256 bytes 에 align 된 메모리를 잡아서 반환 함 
      - 즉, 2D matrix 에서 (row, col) 원소 `주소 계산` 시`,
        - width 대신, `pitch` 를 사용해야 함
        - `T* pElem = (T*)((char*)baseAddr + row * pitch) + col;`
    - `cudaMallocPitch()` 사용하면, `cudaMemcpy2D()` 사용해서 메모리 카피 해라~!
    - 약간 빨라진다고 하는데, RTX4090에서는 약간 느려짐
    - 그런데, 누적되면 빨라진다고하여, 
    - `강의에서는 전체 속도에 영향을 미치기 때문에, 이 테크닉은 꼭 쓰길 권한다고 함`
- 1억개의 2D 행렬 더하기
    - cuda
      - 1349 usec = 0.001349 초
    - pitched
      - 1442 usec = 0.001442 초 
- 10억 개의 2D 행렬 더하기
  - cuda
    - 13,001 usec = 0.013 초
  - pitched
    - 16,191 usec = 0.016 초
- 약간 빨라진다고 하는데, RTX4090에서는 약간 느려짐


[Return Par3 Memory Structure](../README.md)  