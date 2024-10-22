# Search & Sort

[Section34: Linear Search 선형 탐색](./doc/section34.md)
- n 개의 집합(배열 또는 리스트)에서 x 를 찾아라~ 
  - 제일 먼저 나오는거 1개만 찾자!
- CUDA에서 stride 사용해서 좀 빠르게 가속 시키면..
  - 최소한 C 나 C++ 에서 가속된 것 만큼의 속도를 낼 것이다.

[Section35: Search All 모든 위치 모두 찾기](./doc/section35.md)
- n 개의 집합(배열 또는 리스트)에서 모든 x 를 찾아라~
- CUDA에서 stride 사용하는 것이 제일 빠르다. 

[Section36: Binary Search 이진 탐색](./doc/section36.md)
- 주어진 n 개의 element 에서 x 를 찾아라
- binary search `이진 탐색`
  - n 개의 element 는 `정렬`되어 있다고 가정 sorted
- CUDA 사용해서, `binary search`는 효과적이지 못하다.
  - `그냥 CPU 사용하세요!. 특히 STL 짱짱 빠름.`

[Section37: Even Odd Sort 이븐-오드 방식 정렬](./doc/section37.md)
- `CUDA 에서 Sort 하는 방법`.. 본격적으로 얘기해 보자!!
- 블럭 단위 parallel sorting
  - CUDA even-odd sort: 엄청 빨라 짐
- global 메모리 활용 parallel sort 할때는, 
  - CUDA (even-odd) 에서 도차도 상당히 느리다.
  - 개선 방법은.. 아래 섹션에..
  
[Section38: Bitonic Sort 바이토닉 소트](./doc/section38.md)
- bitonic sort
  - 병렬 처리를 위한, 소팅 방법이라고 보면 됨
- block 단위 parallel sort
  - CUDA bitonic sort 를 구현해서, 얼마나 빨라지는 지 확인.
- 대용량 데이터 기반의 bitonic sort 를 위해 global 메모리 사용법에 대해서 알아봄 

[Section39: Counting Merge Sort 카운팅 방식 머지 소트 (병합 정렬)](./doc/section39.md)
- 병렬 처리에 가장 적합한 merge sort 에 대해서 알아보자!
  - Large Scale Parallel Counting Merge Sort
  
[Return Main Readme](../README.md)  


