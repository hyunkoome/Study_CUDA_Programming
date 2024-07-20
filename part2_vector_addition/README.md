# Vector addition

[Section 10: Elapsed Time (소요된 시간 측정)](./doc/section10.md)
- modern C++ 의미
- Wall-clock time vs CPU time
- 시간 측정 방법
  - C++ Chrono features
  - clock function
  - sleep() function
- Variable Arguments (argc: argument count, argv: argument vector 처리)
- 환경 변수 Environment Variables (envp)

[Section 11: CUDA kernel launch 대략적인 설명](./doc/section11.md)
- CUDA Programming Model
- Calling a Kernel Function
- IDs and Dimensions
- CUDA pre-defined data types
- C++ class designs
- Kernel Launch Syntax
- CUDA pre-defined variables
- CUDA Architecture for threads
- thread block queue 의 요구사항

[Section 12: 실제 예제활용 CPU vs CUDA](./doc/section12.md)
- `vec_add_cuda_single_core.cu`
  - CUDA single core 속도 << CPU core 속도
- `vec_add_cuda_error.cu`
  - 우리 GPU에 100만게 core가 없음. 
  - 그런데, 100만개의 코어를 요구해서 돌리려고 함 => 안되죠 !
  - M (streaming multi-processor)에서 1M개의 thread를 동시 실행 불가능 -> 실제로는 1024개가 한계
- `vec_add_cuda_1024_core.cu`
  - vector 크기 SIZE = 1024x1024, 즉, 100만개
  - blockDim = 1024: 쓰레드 블록에는 1024개를 돌리도록 하고, 즉, 블록 1개 안에는 1024개가 돌아가고
  - gridDim = SIZE / 1024: 블록 개수 계산, 100만개를 이 쓰레드 블럭 1개에 1024개가 돌라가니깐, 1M/1024 나눠버리면 => 블록의 개수가 나옴
  - 이제, cpu 보다 빨리 돌아가는 것을 확인 가능
- `vec_add_class.cu`
  - CUDA 커널 함수는 kernel function 
    - `class member 로는 불가능 !!`
    - 그래서, 커널 함수는 외부의 external void 함수로 선언하는 게 일반적임!!
      - `__global__ void` kernelVecAdd(){...}
  - C 코드 보다는, C++ 로 바꾸면, 코드는 동작하지만,
    - C++ 바꾸면 클래스 핸들링하고 하는데, 시간이 조금 더 걸릴 수 있음
    - `확장성 등 고려해서, 코드는 가능하면 C 형태로 사용하는 것이 좋다!`

[Section 13: GIGA 개수 덧셈, CPU vs CUDA](./doc/section13.md)
-`giga_add_host_cpu.cu`
  - 256M 개
    - 362615 usec = 0.36초
    - 에러 0
- `giga_add_cuda_single_core.cu`
  - 12505577 usec = 12.5초
  - 꽤 느리더라
- `giga_add_cuda_single_core.cu`
  - 3596 usec = 0.0036초
  - cpu 보다 빨라진거 확인 함
- `giga_add_clock.cu`
  - 커널 함수 내에서 elapsed time (usec) 계산 방법 
    - elapsed time (usec) = # of clock ticks * 1000.0f / clock frequency (kHz)
- `giga_add_augmentation.cu`
  - 실행시 augment 처리 (SIZE 값 넣을 수 있게..) 

[Section 14: AXPY and FMA](./doc/section14.md)
- AXPY : a X plus Y problem
- FMA : `fused multiply-add` instruction
- LERP: linear interpolation 문제에 적용

[Section 15: thread and GPU](./doc/section15.md)
- CUDA hardware
- Transparent Scalability 확장성 
  - SM 에 들어있는 독립 unit 이 실제로 계산을 위한 processing unit이 됨, 즉 단위가 됨 
  - thread block scheduling
  - `thread block 사용하는 이유에 대해서 구체적으로 설명`
- Thread and Warp
- CUDA thread scheduling
  - wrap sheduling
- Overall execution model
- 실습: warp_lane.cu - warp ID, lane ID check
  - 컴파일: nvcc -o warp_lane warp_lane.cu 
  - cmake 로 컴파일시 에러가 남 (fix 필요!)


