# Vector addition

### CUDA kernel launch

***CUDA Programming Model***
- `parallel code (kernel)` is launched and executed on a device by `many threads`
  - multiple threads → ~10 threads
  - `many threads` → `1,000+ threads` → 실제로는 1M+ threads
- on the `many-core GPUs`
  - multi-core CPU → 10– cores
  - `many-core GPU` → `1,000+ cores` → 실제로는 10K+ cores
- 보통, `thread 개수 >> core 개수`
  - big-size data! → 실제로는 1G ~ 1T items
- many threads on many-core
  - for example, `1,000,000 threads on 1,000 cores`
- launches are hierarchical `계층 구조`: grids → blocks → threads 
  - `Threads` are grouped into `blocks` 여러개 thread -> 1개의 block
  - `Blocks` are grouped into `grids` 여러개 block -> 1개의 grid
  - 1개의 `kernel` 은 1개의 `grid`를 실행하는 구조
- `familiar sequential code is written for a thread`
  - `각 thread 내부는 사실상 sequential code` 순차 처리 모델
  - Built-in thread and block ID variables

***Calling a Kernel Function***
- kernel function 선언
  - `__global__ void` kernel_func( ... ) {...}
- kernel function 호출
  - __host__ function 에서,
    - kernel_func`<<<` dimGrid, dimBlock `>>>`( ... );
      - `dimGrid`: 각각은 그리드 내에 있는 `블록의 개수`
      - `dimBlock`: 하나의 블록에 있는 `쓰레드 개수`
    - kernel_func`<<<` 8, 16 `>>>`( ... );

***IDs and Dimensions***
- grid, block 구조는 `최대 3차원` !
  - 1D: 1차원 배열
  - 2D: 2차원 배열, 행렬 matrix, 영상 image
  - 3D: 3차원 그래픽 자료
  - ID = identification number 식별 번호
- grid: kernel 마다 1개
  - grid dimension: 내부 block 배치
- block: (x,y,z) block index (ID)
  - block dimension: 내부 thread 배치
- thread: (x,y,z) thread index (ID)

***CUDA pre-defined data types***
- Vector types
  - char1, uchar1, short1, ushort1, int1, uint1, long1, ulong1, float1: 1개 묶은 자료형
  - char2, uchar2, short2, ushort2, int2, uint2, long2, ulong2, float2: 2개 묶은 자료형
  - char3, uchar3, short3, ushort3, int3, uint3, long3, ulong3, float3: 3개 묶은 자료형
  - char4, uchar4, short4, ushort4, int4, uint4, long4, ulong4, float4: 4개 묶은 자료형
  - longlong1/2/3/4, ulonglong1/2/3/4, double1/2/3/4
  - `dim3`: 3차원 데이터 자료형  == `uint3` 와 같다
- Components are accessible as `variable.x, variable.y, variable.z, variable.w.`
  - 하나하나 원소 접근시 `.x`, ..., `.z` 사용 
  - we can consider it as a coordinate value: `(x, y, z)` or `(x, y, z, w)`
  - 생성자는 __host__ __device__ `make_float4( x, y, z, w );`
    - 즉, 생성자는 host 프로그램에서 써도 괘찮고, 혹은 디바이스 즉 커널 프로그램에서 써도 문제없이 작동함

***C++ class designs***
- default arguments
  - constructor: dim3( unsigned x = 1, unsigned y = 1, unsigned z = 1 ); 값 생략하면 default = 1임
  - dim3 can take 1, 2, or 3 arguments:
    - dim3 dimBlock1D( 5 ); → (5, 1, 1) : `1차원 정의` 
    - dim3 dimBlock2D( 5, 6 ); → (5, 6, 1) : `2차원 정의`
    - dim3 dimBlock3D( 5, 6, 7 );  `3차원 정의`
- implicit type conversion 암시적 형 변환
  - int1 → dim3 로 자동 변환 가능 : `즉, 아래 3개 모두 같음`
    - kernelFunc <<< 3, 4 >>>( . . . );
    - kernelFunc <<< dim3(3), dim3(4) >>>( . . . );
    - kernelFunc <<< dim3(3,1,1), dim3(4,1,1) >>>( . . . );

***Kernel Launch Syntax***
- kernel function 호출
  - __host__ function 에서.
  - 이미 dimGrid 와 dimBlock 를 선언 후, 사용하는 것이 낫다. 
  - `dim3` dimGrid ( 100, 50, 1 ); // 100 * 50 * 1 = 5000 thread blocks
  - `dim3` dimBlock ( 4, 8, 8 ); // 4 * 8 * 8 = 256 threads per block
  - kernel_func<<< dimGrid, dimBlock >>>( ... ); totally, 5000 * 256 threads !

***CUDA pre-defined variables***
- pre-defined variables
  - dim3 gridDim: dimensions of grid → gridDim.x
  - dim3 blockDim: dimensions of block → blockDim.x
  - uint3 blockIdx: block index within grid → blockIdx.x 내가 속한 block 의 index
  - uint3 threadIdx: thread index within block → threadIdx.x 내가 실행되고 있는 내 쓰래드의 index 
  - int warpSize: number of threads in warp, wrap 내의 thread 개수, 어떤 상수 값 
- 모든 thread 에서 사용 가능, 또, 각 변수 모두 access 할수 있다.
  - gridDim.x, gridDim.y, gridDim.z
  - blockDim.x, blockDim.y, blockDim.z
  - blockIdx.x, blockIdx.y, blockIdx.z
  - threadIdx.x, threadIdx.y, thrreadIdx.z

***CUDA Architecture for threads***
- 1개의 SP (streaming process)는 
  - 기본 unit
  - 1개(single) thread 돌리며,
  - 아주 많은 register 를 할당, 보통 64K 정도
- SM (streaming multi-processor) 스트리밍 멀티 프로세스  
  - 기본 unit: 1개의 SP (streaming process)는 1개(single) thread 돌리며, 
    - 아주 많은 register 를 할당, 보통 64K 정도
  - for a thread block, thread 블록을 돌림
      - 통상, 1개의 thread block 은 1024개 thread 임 
  - SM = a set of SP
    - 보통 32개의 SP
  - ALU's + CU (control unit)
  - `SM의 물리적 한계` = `thread block의 최대 크기`
    - SM은 동시에 32개의 thread 를 실행하지만, 실제로는 1024개 threads를 갖고 있으므로,
    - 1024개의 thread 중 대부분은 대기
  - `→ (정리)` 
    - 일종의 하나의 CPU처럼 동작 하여 
    - 실제 실행은 32개의 쓰레드를 동시에 실행 시킬 수 있지만 (time sharing !)
    - 대기 상태에는 무려 1000개가 대기하고 있다가,
    - 저 32개가 잠시 멈칫하면, 즉시, 다른 32개가 실행되고
    - 또 그 32개가 멈칫하면, 다음 32개가 실행되면서, 
    - 전체 SM 관점에서는 최대로 돌아감 

***thread block queue 의 요구사항***
- 쓰레드 블럭들을 저장
- `하나씩 가져가서, 실행하고, 제거`
- 정확한 우선순위가 필요한가? 또는 정확한 우선수위를 계산가능한가?
- 느슨한 queue-like 자료 구조로 관리해도 충분 


[Return Par2 Vector Addition](../README.md)  