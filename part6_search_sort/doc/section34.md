# Linear Search

***linear search***
- n 개의 집합(배열 또는 리스트)에서 x 를 찾아라~
- 정렬 되어있지 않다고 가정 unsorted
- for-loop 로 구현 가능: O(n) 알고리즘
- 제일 먼저 나오는 1개만 찾자!

***find the first occurrence for 16M elements***
- 문제 sample problem
  - 총 16M elements
  - unsigned num = 16 * 1024 * 1024;
  - 데이터 한계는 1,000,000 로 설정
  - unsigned bound = 1000 * 1000;
  - 즉, 각 element 는 0 ~ 999,999 까지 값을 가질 수 있음
- target value
  - 16M 개 중에서 제일 마지막 값 사용
  - targetValue = vecData[num - 1];
  - 확률적으로, num / bound = 16.77 개 정도의 target value 가 존재
  - 제일 먼저 나오는 1개만 찾자
- 16M 개의 배열을 가지고, 원하는 값의, 제일 먼저 나오는 값의 위치 찾기
  - host version, 단순 방법 `search_host_naive.cu`
    - 실행 시간: 313 usec
  - C++ find, find 함수(linear search로 내부적으로 구현되어 있음)를 이용한 탐색 `search_host_find.cu`
    - 실행 시간: 484 usec 
      - 원래는 `search_host_naive.cu` 보다 빨라야 하는데, 왜 느린지 모르겠음
  - CUDA atomic `search_cuda_atomic.cu`
    - 쓰래드 1개가 1개의 element를 검사하는 방식
      - if found, uodate index: atomicMin() 
    - 실행 시간: 194 usec 
      - CUDA는 쓰래드 돌려서, 모든 elements 가 다돌지만, 
      - c++에서는 linear sort 로직으로 인해, 모든 원소를 검색하지 않아서,
      - 원래는 `cpu 사용한 것들보다` 보다 느려진다고 하지만.. 
      - 나의 경우는.. 더 빠름.
  - CUDA block `search_cuda_block.cu`
    - 쓰래드 1개가 count 개의 element를 검사하는 방식
      - 이 예제에서는, count 는 1024개
    - 실행 시간: 537 usec
      - CUDA는 쓰래드 돌려서, 모든 elements 가 다돌지만,
      - c++에서는 linear sort 로직으로 인해, 모든 원소를 검색하지 않아서,
      - `cpu 사용한 것들보다` 보다 느려짐
      - 그리고, memory coalescing 문제로 메모리를 제대로 못쓰고 있어서.. 엄청 느려짐.
  - CUDA stride `search_cuda_stride.cu`
    - CUDA block 의 memory coalescing 문제를 해결하기 위해..
    - 쓰래드 1개가 count 의 element 를 검사 => stride 사용
    - 실행 시간: 141 usec
      - CUDA에서 stride 사용해서 좀 빠르게 가속 시키면..
      - 최소한 C 나 C++ 에서 가속된 것 만큼의 속도를 낼 것이다.

[Return Par6 Search & Sort](../README.md)  