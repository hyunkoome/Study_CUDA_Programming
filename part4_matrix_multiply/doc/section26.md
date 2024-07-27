# Memory Structure

***신형 CUDA device 개선점***
- SM: streaming multiprocessor
  - 최신 구조는 `SM 내에 독립 unit` 이 2개, 그 이상 일 수도 있음
    - 1 unit = 32 SP + 16 DP + 8 SFU 로 구성
    - `SP`: streaming processor = core
    - `DP`: double precision
      - 더 정밀도 높은 floating point 숫자 연산 유닛
    - `SFU`: special function unit 
      - cos, sin 같이 복잡한 함수들을 굉장히 빨리 처리할 수 있는. 유닛
  - `Shared Memory 와 L1 cache 포함` 


***Programmer View of CUDA Memories***
- Each thread can:
  - read/write per-thread `registers`
  - read/write per-block `shared memory`
  - read/write per-thread `local memory`
    - actually, located on the `global memory` <== `예전에는 local = global 메모리였는데,`
    - now, `on the L1 cache !` <== `지금은 새로운 CPU에서는, local = L1 cache로..!!`
      - 그러나, L1 cache 가 가득차면, global memory 로 보내서, 용량을 키워서 사용할 수 있게 해줌!
  - read/write per-grid `global memory`
  - `read-only` per-grid `constant memory`


***변수들 실제 스피드 측정 (`speed_variables.cu`)***
- core 1개 사용 => 1개의 쓰래드만 실행해서, 1개의 코어만 사용해서, 스피드 계산
- 1024 x 1024 개의 더하기 = 1M (밀리언) 개 덧셈
- 쉐어드 메모리: 8 x 1024 엘리먼트 = 8K 개의 엘리먼트, 
  - => floating point 사용 => 개당 4 바이트(B) 필요!
  - => 8K x 4 B = 32 KB 용량의 shared 메모리 사용함
  - 대부분, 우리사 사용하는 GPU의 shared 메모리 용량이 최대 48KB 정도이므로,
  - 이 정도 32 KB 정도 잡는게 안전 함
- 연산 시간
  - register 변수:         4,653,833 ticks
  - shared memory 변수:    7,215,134 ticks
  - global memory 변수:   90,934,124 ticks
  - local memory 변수:     7,554,878 ticks
  - constant memory 변수:  4,653,790 ticks 
    - constant 는 읽기용으로만 사용가능해서, register 변수 하나 추가해서 
    - register 변수에 constant 더하는 식으로 ..
    - 그래서, 연산시간은.. 절반은 register 사용한 시간이고, 절반은 constant 변수 사용한 시간으로 봐야 함!
- `의외로, constant 변수를 사용한 경우가 빠름`
  - `read only 만 하는 경우라면, constant 변수를 꼭 사용해야 함!!`
- 예상한 대로, `register variable > shared variable > global variable 순서로 빠르다.`
- `예상 외로, local variable 이 상당히 빠름 -> 이제 L1 cache에 위치` (CUDA 6.0 이후)

***추가 고려 사항들***
- memory clear (보통 global 메모리)시 속도

[Return Par4 Matrix Multiply](../README.md)  