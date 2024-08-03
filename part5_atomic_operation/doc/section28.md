# Control Flow Overview

- SM architecture
  - 1 CU + 32 cores
- if 문 과 for loop 문 어떻게 최적화 할것인지?
  - divergent cases
    - if statements
    - for-loop iterations
- 숫자를 서로 섞어야 하는 예제에서..even-odd 와 half-by-half 어떻게 구현하는지, 그리고 수행 시간..
  - Example: shuffling cases 
    - even-odd
    - half-by-half
    - even-odd, shared mem
    - half-by-half, shared mem

# Control Flow
- 요즘 GPU는 보통 2개의 unit으로 구성되며,
- 1개의 unit 안에는 32개의 SP (= FP32 core), 8개의 SFU(수학함수 등 연산 유닛), 1개의 CU (컨트롤 유닛) 으로 구성
- 그런데, 문제는 CU (= warp 와프 스케쥴러)는 1개뿐임.
  - 1개의 warp 는 32개의 쓰래드로 구성되어 있고, CU 1개로 32개 쓰래드를 동시에 병렬로 control 해야 함
  - 1개의 warp 내에서, 서로 다른 path를 수행해야 하는 경우(divergent branches: if-else, for-loop) 에
  - CU 1개로 수행하다보니, 어쩔수 없이 serialized (순차 수행) 
  - 서로다른 warp 에서 서로다른 path를 수행하는 것은 performance에 영향이 없음
- algorithm-level optimization (if 문)
  - even-odd case
    - warp 내 모든 쓰래드가
    - if-part, else-part를 모두 수행
    - 속도가 좀 느려질 수도 있음 
  - half-by-half case
    - warp 전체가 if-part 또는 else-part 중 1개만 수행
    - 속도면에서, even-odd case 보다 이득을 봄 
- algorithm-level optimization (for-loop)
  - for 돌아야 하는 루프의 최대로 잡아서.. 돌림 
- Example: Shuffling Problem
  - even numbered items → left half
    - 각 thread 는 source 입장에서 해석
      - even-numbered threads: `dst[gx / 2] = src[gx];` 
      - odd-numbered threads: `dst[HALF + gx / 2] = src[gx];`
      ```c++
      if (gx % 2 == 0) 
      {
        dst[gx / 2] = src[gx];
      }
      else 
      {
        dst[HALF + gx / 2] = src[gx];
      }
      ```
  - odd numbered items → right half
    - 각 thread 는 destination 입장에서 해석 
      - left-half threads: `dst[gx] = src[gx * 2];`
      - right-half threads: `dst[gx] = src[(gx – HALF) * 2 + 1];`
      ```c++
      if (gx < HALF) 
      { // left half
          dst[gx] = src[gx * 2];
      } 
      else 
      {
          dst[gx] = src[(gx - HALF) * 2 + 1];
      }
      ```
  - `evenodd.cu`
    - 연산 시간: 2,483 usec
  - `halfhalf.cu`
    - 연산 시간: 3,564 usec
    - evenodd 케이스가 더 느릴거라고 생각했는데, 
      - 의외로 half-and-half 쪽이, 컨트롤 플로우 입장에서는 더 적은 일을 했는데도,
      - 수행시간은 더 걸려버린 상황이 생겼음
    - 원인 분석
      - global memory 접근 횟수의 차이
        - even-odd 케이스: memory access 에 겹치는 부분이 없음
        - half-by-half 케이스: read access 에서 중복 발생, 안그래도 global 메모리 읽을때 느린데..
          - control flow 최적화 해도, 메모리를 2번 읽어오는데, 빨라질수 없음. even-odd 케이스 더 빠름
- 근데 왜? CUDA 프로그래머들은 `half-by-half`를 `선호`할까?? 
  - ***shared 메모리를 사용하면 half-by-half 를 사용해도, 메모리를 1번 읽어오기때문에, half-by-half가 빠름.***
  - shared memory에서 shared memory로 정리할때, `evenodd` 와 `half-half`를 사용할 예정  
  - `evenodd_shared.cu`
    - 연산 시간: 2,527 usec
    - shared memory 사용하니, 이전에 even-odd 하던 것 보다는 느려질 수 밖에 없고.
    - 그래도, half-and-half 보다는 빠르다.
  - `halfhalf_shared.cu`
    - 연산 시간: 2,484 usec
    - `shared memory` 에서는 확실히.. `half-and-half` 가 빠르다!! 그래서..선호~~

# Stride 보폭 정의
- stride : offset between adjacent data elements
  - 기준 단위: byte 가 아니라, CPU/GPU word (4바이트) 기준
- 메모리 효율적인 면과, CUDA 속도면에서, SoA 가 AoS 보다 우수함 
  - Array of Structures (AoS)
    - ```c++
      struct recordA {
        int key; // red
        int value; // green
        int flag; // blue
      };
      recordA recA[100];
      ```
    - 각 데이터를 읽으려면, stride 3로 처리하는 꼴임        
  - Structure of Arrays (SoA)
    - ```c++
        struct recordB {
          int key[100]; // red
          int value[100]; // green
          int flag[100]; // blue
        };
        recordB recB;
        ```
    - 각 데이터를 읽으려면, stride 1로 처리할 수 있어서. 한번에 처리 가능
  - 따라서, 최적화 시에는 반드시 global memory access 패턴의 분석이 필요함!
  

[Return Par5 Atomic Operation](../README.md)  