# Binary Search

***binary search problem***
- search problem 탐색 문제
  - 주어진 n 개의 element 에서 x 를 찾아라
- binary search 이진 탐색
  - n 개의 element 는 `정렬`되어 있다고 가정 sorted
  - element 들은 `배열 array 에 저장`되어 있다
  - O(log n) 알고리즘
    - `sort 된 array 가 필요`
    - recursive version: find an element in [first, last)


***find the first occurrence for 16M sorted elements***
- host naïve version
  - 실행 시간: 11 usec
- C bsearch `stdlib.h 포함된 bsearch() 사용` 
  - 실행 시간: 11 usec
- C++ binary_search() (`stl 사용, #include<algorithm>`)
  - 실행 시간: 4 usec
- CUDA kernel  
  - 실행 시간: 178 usec
    - CUDA는 쓰래드 돌려서, 모든 elements 가 다돌지만, 
    - c++에서는 linear sort 로직으로 인해, 모든 원소를 검색하지 않아서,
    - 원래는 `cpu 사용한 것들보다` 보다 `CUDA` 가 느려진다

[Return Par6 Search & Sort](../README.md)  