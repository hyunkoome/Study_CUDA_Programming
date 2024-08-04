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
  - C++ find, find 함수를 이용한 탐색 `search_host_find.cu`
    - 실행 시간: 484 usec 
  - CUDA atomic `search_cuda_atomic.cu`
    - 실행 시간: 194 usec 
  - CUDA block `search_cuda_block.cu`
    - 실행 시간: 537 usec
  - CUDA stride `search_cuda_stride.cu`
    - 실행 시간: 251 usec

[Return Par6 Search & Sort](../README.md)  