# Even Odd Search: 본격적으로, CUDA에서 Sort 하는 방법

***Sorting Algorithms***
- two categories of sorting
  - internal sort: fits in memory
  - external sort: uses auxiliary storage <-- 병렬 처리..에 필요 
- comparison based sorting <-- 일반적으로 정렬이라고 함은.. 여기 임
  - compare-exchange
  - quick sort, merge sort, ...
  - O(n log n)
- non-comparison based sorting 
    - uses known properties of the elements
  - O(n) - bucket sort, radix sort, etc.

***parallel sorting 문제***
- Input and Output sequence storage
  - Where?
  - Local to one processor or distributed
    - CUDA : stored in global memory / shared memory
    - other distributed environment: distributed cases
  - Comparisons
    - How to compare elements on different nodes
      - CUDA: how to compare elements on different TB (thread blocks)
  - # of elements per processor
    - CUDA: # elements per thread
    - only one (compare-exchange → communication)
    - multiple (compare-split → communication)

***Compare-Exchange***
- one element per thread
  - data on each node / thread
  - send the data to each other
  - compare and then, get the new data
- multiple elements per thread <== `블럭 단위로 sorting : CUDA의 기본 아이디어` 
  - list of elements on each node / thread
  - combine the lists, then split it

***Block 단위 sort 의 구현***
- a thread block
  - 최대 1,024 threads
  - shared memory size: 48KB 최대, 엘리멘트가 최대 12K 개 integers/floats
  - block 1 개의 sort 에 집중
  - 효율성을 높이기 위해, 2 x 1,024 = 2,048 개를 sort 하자
- a big-size problem, as the next step
  - thread block 단위 sort 를 `기본 연산`으로 쓰자
  - 최종적으로는 1M 이상을 sort 하자 
  
***block 단위 parallel sort***
- qsort in C `qsort_block.cu`
  - 연산 시간: 7,012 usec
- std::sort in C++ `std_sort_block.cu`
  - 연산 시간: 4,746 usec 
- bubble sort `cpu_bubble_block.cu`
  - 연산 시간: 65,634 usec 
  - 진짜로 느린 알고리즘 임.
- even-odd sort `cpu_even_odd_block.cu`
  - bubble sort 의 변형 => 병렬 처리가 가능해 짐 
  - even-odd sort
    - first phase: even numbered pair compare-and-swap
    - second phase: odd numbered pair compare-and-swap
  - 연산 시간: 65,004 usec
    - 병렬처리 안하면, 속도는 별 차이 없음
    - 그러나, 병렬 처리 하는 법을 배웠으니, CUDA 로 돌리면, 빨라질 것임.
- CUDA even-odd sort `even_odd_block.cu`
  - block 단위 sort
    - `shared memory`
  - 2,048 elements → fit to the shared memory
    - 1,024 threads → 각 쓰래드 당 2개의 elements 로 시작
    - read data[i] and data[i + 1024]
  - even-odd sort kernel
    - phase 1: swap data[2*i] and data[2*i + 1] 
      - => (0,1) (2,3), ... even numbered pair
    - phase 2: swap data[2*i – 1] and data[2*i] 
      - => (1,2) (3,4), ... odd numbered pair
  - 역 방향도 가능하게 하자
    - 0 = decreasing order, 1 = increasing order
  - 연산 시간: 302 usec

***global 메모리 활용 parallel sort***
- block 단위 sort 를 벗어나는 경우의 처리는 어떻게 할 것인지?
  - 즉, shared memory를 이용 못하는 경우, 
  - 즉, 100만개 는 정렬 어떻게 하노?
    - => `global memory 에 대해서 직접 even-odd를 적용` 하자!  
- CUDA even-odd sort `even_odd_global.cu`
  - 연산 시간: 299 usec
- CUDA even-outer bubble `even_odd_outer_bubble.cu`
  - even-odd sort 를 2단계로 해보자.
    - 연산 시간: 8,086 usec
    - global 메모리.. 느리긴 하네..!!


[Return Par6 Search & Sort](../README.md)  