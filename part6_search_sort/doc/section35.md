# Search All 전수 탐색: 모든 위치, 모두 찾기

***linear search, all elements***
- n 개의 집합(배열 또는 리스트)에서 `모든` x 를 찾아라~
- 정렬 되어있지 않다고 가정 unsorted
- for-loop 로 구현 가능: O(n) 알고리즘

***find all occurrences for 16M elements***
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
    - `모두 다 찾자 => 찾은 indx 들을 보관하는 배열 필요!`
- 16M 개의 배열을 가지고, 원하는 값의, `모든` 위치 찾기
  - host version
    - 실행 시간: 10,647 usec
  - C++ find
    - 실행 시간: 14,917 usec 
  - CUDA atomic
    - if found, update index: atomicADD()
      - index를 업데이트해서, 전체 저장하고 싶은 배열의 어느 위치에 넣어야 될지를 다시 계산
    - 실행 시간: 190 usec
  - CUDA block
    - 실행 시간: 534 usec
  - CUDA stride
    - 실행 시간: 180 usec
  - CUDA stride2     
    - atomoic op 안쓰는 버전.
      - 가정: 쓰래드 1개가 최대 1개를 찾을 것이다. 
    - 실행 시간: 203 usec

[Return Par6 Search & Sort](../README.md)  