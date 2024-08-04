# Counting Merge Search 카운팅 방식 머지 소트 (병합 정렬)
- 병렬처리에 가장 적합한 merge sort 에 대해서 알아보자!

***Counting Merge Sort***
- basic idea: 각 element 가 자기 위치를 따로 계산하자!
- for an element x in the sub-list A
  - sub-list A: 내 위치가 pos A 번째 → 내 앞에 pos A 개가 있다.   O(1)
  - sub-list B: 
    - 나보다 작은 갯수 count → 내 앞에 pos B 개가 있다.           O(n)
    - => binary search 로 (>= x) 값의 위치 pos B 를 찾는다.    O(log n)        
  - combined list C: 내 위치는 (pos A + pos B ) 이다 !
- for an element x in the sub-list B
  - sub-list A: binary search , (> x) 값의 위치 pos B를 찾는다 -> 내 앞에 Pos B 개 가 있다.
  - 같은 값을 가지는 경우의 처리 가능
- 장점: 각 element 별로 완전한 병렬 처리 가능!
  - O((log n)^2) with n processors

***Large Scale Parallel Counting Merge Sort***
- merge sort 의 본질은?
  - (binary) tree 형태로 big-size block들을 계속 합쳐야 한다
  - 다만, 이제 big size 로!!
- parallel merge sort, again
  - 2 개의 block을 merge해서 1 개의 큰 block으로 만든다
  - CUDA 에서의 `limit`: shared memory size, # of threads in a block
    - 한번에 1,024 개 정도만 merge 가능 
- big-size block ( 또는 segment)
  - traditional merge: 1 개의 processor 가 순차 처리
  - 새로운 idea: `CUDA 에서 처리가능한 size로 분할 하자`
    - (최대 block size) / 2 정도의 sub-block 들로 분할해서 처리하자
    - sampling point 1개의 경우: 2-way parallel processing
    - sampling point `n개`의 경우: n-way parallel processing
    - sampling point 의 선택?
      - merge buffer 크기를 shared memory 크기보다 작게, 되도록 규칙적으로
    - `rank`: `512 개 마다` sampling point, `최대 합쳐지는 크기는 1,024`
      - => 쓰래드 블럭 하나로 처리 가능
    - `limits`: 모든 rank는 sorted list
      - 조각난 sub-block 들을 merge !
- `정리` 해보면, 
  - counting merge sort를 global memory에 있는 리스트 전체에 적용
  - step 0: sort `each block` (a TB with 2,048 elements, in our case)
    - we know that `BLOCK_SIZE = 1,024`
  - `repeatedly` merge two blocks into one 
    - sample values : pick every `512` th element
      - let `SAMPLE_STRIDE = BLOCK_SIZE / 2`
    - step 1: get `ranks`: binary search for every sample values
    - step 2: get `limits`: sort the ranks, to get the limit index for every `sub-blocks`
    - step 3: merge: `thread-block` 에서 `parallel merge` !

***실행 시간 확인***
- counting merge sort
  - block 단위 처리 `cpu_merge_block.cu`
    - 실행 시간: 4,650 usec
  - big-size merge `merge_increasing_block.cu`
    - 실행 시간: 48 usec 
- block 단위 parallel sort
  - CUDA bitonic sort `merge_block.cu`
    - 실행 시간: 60 usec
  - CUDA parallel merge `merge_outer_bubble.cu` 
    - 블럭 단위 버블 소트 적용
    - 실행 시간: 316 usec 
- global 메모리 사용시, parallel sort
  - CUDA bitonoc sort `merge_outer_bubble_large.cu` (1M 개)
    - 실행 시간: 36,365 usec
  - CUDA 글로벌 카운팅 머지 소트 `merge_global.cu` (32,768 개)
    - 실행 시간: 479 usec
  - CUDA global merge `merge_global_large.cu` (1M 개)
    - 실행 시간: 4,666 usec

[Return Par6 Search & Sort](../README.md)  