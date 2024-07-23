# Matrix Multiply 행렬 곱셈

[Section22: matrix copy](./doc/section22.md)
- CPU 사용
  - CPU에서 이중 루프 돌린거 보다 메모리 카피 쓴게 의외로 약간 느림
  - 이 결과는 메모리 카피가 운영체제에서 지원 되면서, 굉장히 효과적으로 짜여있긴하지만,
  - 경우에 따라서는 cashe 상황이나, 혹은 컴파일러가 for 루프를 얼마나 optimize 했느냐에 따라,
  - 이런 관계 때문에, `경우에 따라서는 우리가 for 루프 돌려서 직접 짠 것이 더 빠를수도 있다는 것을 보여줌`
- CUDA naive version – `global memory`
  - naive : (전문적) 지식이 없는
  - `pitched matrix` 를 사용하는 것이 속도가 빠르므로.. 이 방법을 사용
  - 분명히, cpu 버전들 보다 빨라짐
- CUDA `shared memory` 사용: `tiled approach` 사용 해야함
  - 글로벌 메모리 사용 버전과 비교하면, 다소 느리지만,
  - 그 이유는 shared 메모리를 한번 더 거쳤으니까, 느릴수 밖에 없다.
  - `그러나, 일반적으로 shared 메모리에서 행렬 copy가 아닌, 실제 연산등이 추가되므로.`
  - `이 방법을 사용하는 것을 연습하기를 권장 함`
- CUDA memcpy2D 쿠다 커널 직접 카피 (`cudaMemcpy2D()`)
  - CUDA에서 글로벌 메모리 쓰는 버전이나, 쉐어드 메모리 쓰는 버전보다. 빨라짐
  - 그래서, `사실은 이것이.. 우리가 CUDA 프로그래밍할때 기준이 되는, 속도 레퍼런스가 됨`
    - 이것 보다 더 빠른 알고리즘을 우리가 CUDA에 구현하는 것은 불가능함.
  - the best score 최고 기록 for any matrix operations 행렬 대입 연산이 최고 기록이다.
    - 따라서 다른 연산, 즉 더하기 연산 등은 이 행렬 대입연산의 속도에 근접할 수록 최적화가 잘 된것이고
    - 이 기록을 추월했다면, 만세를 부를게 아니라, 뭔가 잘 못된 것임
    - 즉, 연산의 물체를 발견할 수 있는 어떤 기준이 될수도 있음
  
[Return Main Readme](../README.md)  


