# Vector addition

### thread and GPU

***Transparent Scalability 확장성***
- SM 에 들어있는 독립 unit 이 실제로 계산을 위한 processing unit이 됨, 즉 단위가 됨
- `CUDA device 는 매우 다양할 수 있음`
  - 저가 스마트폰, 태블릿: 1~4개의 SM만 있는 저가 GPU 채용
  - 일반적인 PC: 6~200개의 SM을 가진 고가 GPU 채용
  - high-end 워크스테이션: 1000+개의 SM 가능
- CUDA가 제시한 해결책 solution
  - `thread block 개념의 도입`
  - SM 1개가 thread block 1개를 처리
  - `grid - block - thread` 의 계층 구조 (hierarchy) 가 필요한 이유 
- Nvidia Tesla GP100 GPU 예를 들면
  - SM: streaming multiprocessor
  - 1개의 CU (control unit)
    - 32개 core = 32개 thread 가 동시 실행 
    - 1개의 warp scheduler ..       
    - 32개 쓰래드 -> 같은 instruction을 동시 실행
  - time sharing 관점
    - SM 1개는 2048+thread를 동시 관리
    - memory의 느린 반응 속도 해결

***Thread and Warp***
- a single thread: 독립적 실행의 단위
- warp = 32개 threads
  - 물리적으로 (진짜) 동시 실행되는 thread 그룹
- lane
  - warp 내에서 각 thread의 index
  - 0 ~ 31

***CUDA Thread Block***
- Programmer declares (thread) Block: 
  - Block can have 1 to 1024 concurrent threads
  - Block shapes 1D, 2D, or 3D
- A single kernel program !
  - All threads execute `the same kernel program`
  - Threads have `thread ID numbers` within Block
  - Thread program uses thread ID to select work and address shared data
- Threads run concurrently
  - SM assigns/maintains thread ID's
  - SM manages/schedules thread execution
- `Each Thread Blocks` is divided in `32-thread Warps`: 1 warp = 32 threads
  - This is an implementation decision, not part of the CUDA programming model.
- `1 thread block = 1024 threads` => `1 warp = 32 threads` 
  - 즉. thread block 은 warp 단위로 관리됨 
  - SM안에 32개 코어(32개의 SP)가 있고, 이 32개 코어가 동시에 돌아감
    - clock cycles
      - 1 clock cycle에 1개의 register instruction 실행
      - 메모리 access를 위해서, 100 clock cycles 필요 
      - 평균적으로 4개의 instruction (+, -, x, 나누기 등) 수행하면 1번 메모리 엑세스를 한다.
        - 그럼 100 / 4 = 25 warps 는 대기상태에 있어야. 
        - 1개의 warp 가 메모리 엑세스로 대기일때, 다른 warp들을 계속 실행시킬수 있는 거지.
        - 그래서, 20개 이상의 warp가 필요하다는게. 이런 의미 임
  - 하나의 control unit의 제어를 받아서, 32개의 core가 동시에 돌아가니까,
  - 여기에 32개의 thread를 주는게 자연스럽습니다.
  - 그래서 굳이 thread들을 32개의 thread, 즉, warp 단위로 잘라서 사용하는 것임 
- CUDA는 보면, 나중에 메모리하고 관계를 생각해보면, 
  - 하나의 control unit이 최소한 20개 정도 혹은 그이상의 warp는 동시에 돌리는게 좋음
  - 안그러면, 메모리 접근하는 시간이 오래 걸려서, CUDA 사용이 느려질 수도 있어서
  - 항상 20개 이상은 돌리는게, CUDA 성능이 최고로 올라가더라..

***warp id, lane id***
- 1 warp = 32 threads
  - lane: `warp` 내에서 각 thread 의 index (0~31)
  - thread index 와는 다름: `thread index` 는 `블럭` 내에서 각 thread 의 index  
- warp id: SM 내에서 특정 `warp의 ID number`
  - SM 단위로 관리 -> globally unique 하지는 않음 (다른 SM에 동일 warp id 가능)
  - warp id를 가져오는 함수는 `없음`
  - GPU assembly instruction 으로 체크 가능
- lane id: 1개 `warp 내에서`, `자신의 lane ID number`
  - 마찬가지로, lane id를 가져오는 함수는 `없음`
  - GPU assembly instruction 으로 체크 가능

[Return Par2 Vector Addition](../README.md)  