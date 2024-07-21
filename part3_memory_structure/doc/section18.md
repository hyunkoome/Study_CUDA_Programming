# Memory Structure

### 3D filter 적용 
- `__host__ cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);`
  - pitch를 가지도록 내가 원하는 3D 크기로 memory를 allocate 할당해라
  - `__host__ cudaExtent make_cudaExtent( size_t w, size_t h, size_t d );`
    - make a CUDA extent
      - - make_cudaExtent 는 width, height, depth를 정해주면, 3d 행렬을 만들어줌
    - w : width in bytes 바이트 단위
    - h : height in elements 개수 단위 
    - d : depth in elements 개수 단위 
  - 반환 값 `pitchedDevPtr` 포인터   
    - struct cudaPitchedPtr {
      size_t pitch; // pitch in bytes
      void* ptr; // real pointer
      size_t xsize; // width in bytes
      size_t ysize; // height in elements
      };

***3D Pitched Matrix 함수들*** 
- `__host__ cudaPitchedPtr make_cudaPitchedPtr( void* d, size_t p, size_t xsz, size_t ysz );`
  - returns a cudaPitchedPtr based on input parameters.
  - d : pointer to allocated memory
  - p : pitch of allocated memory in bytes
  - xsz : width of allocation in bytes
  - ysz : height of allocation in elements
    - 이미 만들어진 ptr 로 부터 pitched ptr을 만들어 줌 
- `__host__ cudaPos make_cudaPos( size_t x, size_t y, size_t z );`
  - return a cudaPos, based on the input parameters (x, y, z).
  - 3D array 내에서 특정 위치를 지정
- `__host__ cudaError_t cudaMemcpy3D( const cudaMemcpy3DParams* p );`
  - cudaMemcpy3DParams
    - `cudaPos` srcPos, `cudaPitchedPtr` srcPtr : source location
    - `cudaPos` dstPos, `cudaPitchedPtr` dstPtr : destination location
    - `cudaExtent` extent : 복사할 크기
    - `cudaMemcpyKind` kind : host-device 사이의 방향 설정
  
- 1억개의 3D 필더링
  - cpu
    - 43,814 usec = 0.43초 
  - cuda
    - 348 usec = 0.000348 초
  - pitched
    - 429 usec = 0.000429 초
  - 간단한 연산의 경우에는 별로 이득을 보지 못할 수도 있다!!
  - 복잡한 연산에서는 그대로 여전히 pitched memory 를 사용하는게 더 좋을 수도 있어서, 추천함 !!
  - 그래서, pitched memory 사용을 자꾸 강조함!!!


[Return Par3 Memory Structure](../README.md)  