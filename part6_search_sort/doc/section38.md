# Bitonic Search

***bitonic sort***
- merge sort 와 유사하지만, 다름.
  - single cpu에서는 merge sort가 bitonic sort 보다 훨씬 빠름
  - 그러나, n개의 쓰래드, n개의 cpu, n 개의 gpu 에서 병렬처리하게되면,
    - bitonic sort 가 훨씬 빨라짐 
- bitonic sequence 를 정의한 다음에,
  - bitonic sequence를 arrange 해서 sorting 하는 기법
- bitonic sequence 란?
  - mono-tonic 모노톤 시퀀스: 한 가지 톤, 즉, 모두 증가하거나, 모두 감소하거나
  - bi-tonic 바이-토닉 시퀀스: 2가지 톤을 모두 갖고있는 것.
    - 예1: [2 3 6 8 7 5 4 1] 앞 부분은 증가 하고, 뒷 부분은 감소
    - 예2: [7 5 4 1 2 3 6 8 ] 반대로, 앞 부분은 감소 하고, 뒷 부분은 증가

***Rearranging a Bitonic Sequence***
- unsorted n elements → a bitonic sequence of n elements.
  - two elements = a bitonic sequence
  - merge small bitonic sequences into larger bitonic sequences
    - you need bitonic sort to merge them !
  - until we end up with a bitonic sequence of size n
- bitonic sort: a bitonic sequence of n elements → sorted n elements
  - split it into two bitonic sequences of size n/2
  - apply bitonic sort to two bitonic sequences of size n/2
  - concatenating them into a large sorted list of size n
- divide-and-conquer approach


***large size***
- 바이오닉 소팅을 글로벌 매모리로 확장 
- 32K elements in global memory → 16 x (2K elements in a block)
  - block 1 bitonic sort
- 16 개의 block 은 어떻게 sort?
  - sort 가 아니라 서로 data를 교환하게 하자
  - 블럭을 엇갈리게 배치해서 데이터 교환
  - 전체는 even-odd sort 와 유사



***block 단위 parallel sort***
- std::sort in C++ `cpu_bitonic_block.cu`
  - recursion 방식
  - 실행 시간: 8,084 usec
- bitonic sort `bitonic_block.cu`
  - 실행 시간: 187 usec 
- bitonic sort `cpu_bitonic2_block.cu`
  - 실행 시간: 7,709 usec
  - index 방식
    - data 교환 과정을 index numbering 으로 표현 가능
    - 각 step 에서 어느것과 어느 것을 compare 하는지, index 계산 수식을 찾아냄
- bitonic sort `bitonic2_block.cu`
  - CUDA의 block 단위 구현에도 적용, index 만 써서, bitonic sort 구현
  - 실행 시간: 68 usec
- CUDA bitonic sort `bitonic_outer_bubble.cu`
  - 글로벌 메모리로 확장
  - 실행 시간: 390 usec (for 32,768 엘리먼트)
    - 글로벌 메모리 사용 하지 않을때보다, 글로벌 매모리를 사용하면 5배 이상 느림..
    - 그래도, 글로벌 메모리를 사용하게 됬으니, 큰 데이터도 정렬 가능하고, cpu 보다는 훨씬 빠름
  
***global parallel sort***
- CUDA bitonoc sort `bitonic_outer_bubble_large.cu`
  - 실행 시간: 47,330 usec (for 1,048,576: 1M 개 엘리먼트) 
   
  
  [Return Par6 Search & Sort](../README.md)  