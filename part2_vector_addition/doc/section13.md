# Vector addition

### 실제 예제활용 CPU vs CUDA

***`giga_add_host_cpu.cu`***
- 256M 개
- 362615 usec = 0.36초
- 에러 0

***`giga_add_cuda_single_core.cu`***
- 12505577 usec = 12.5초
- 꽤 느리더라

***`giga_add_cuda_single_core.cu`***
- 3596 usec = 0.0036초
- cpu 보다 빨라진거 확인 함

***`giga_add_clock.cu`***
- clock() in 커널 함수 활용
  - clock_t clock(void); 
  - long long int clock64(void); `64비트`
    - returns the value of a per-multiprocessor counter (or clock ticks)
    - CAUTION: executed in __device__ and __global__ functions 
  - 실제로는, 앞에 __device__ 나 __global__ 이 붙은 쿠다 커널 함수 또는 CUDA의 디바이스 함수 내에서 돌릴 수 있음
  - 즉, cpu 가 아니라, CUDA 내에서 clock tick 이 얼마나 되었는지 알려줌 
  - 그래서, 몇 클럭 만에 수행됬는지, 클럭 획수를 알려주는 식으로 되어 있음 
  - 그런데, 이것을 메인함수에 어떻게 알려주느냐? 
    - 그래서, 64bit 정수를 저장할수 있는 배열(디바이스에 있는 배열)을 하나 더 주고,
    - 커널 함수 파라미터로 long long* times 를 추가..
    - i 번째 실행시간을 times[i]에 클럭 단위로 넣도록 이렇게 구현
- elapsed time (usec) = # of clock ticks * 1000.0f / clock frequency (kHz)

***giga_add_augmentation***
```shell
./giga_add_augmentation # default 값 사용시 
./giga_add_augmentation SIZE
./giga_add_augmentation 268435456
./giga_add_augmentation 512000000
./giga_add_augmentation 8000000000
```
- `out of memory` for 8,000,000,000 elements (80억개) on RTX4090 

[Return Par2 Vector Addition](../README.md)  