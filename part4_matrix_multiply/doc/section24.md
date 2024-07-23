# Memory Structure

***matrix copy – theoretical limit***
- CPU version
- memcpy version
- CUDA naive version – global memory
  - naive : (전문적) 지식이 없는
- CUDA shared memory version
- CUDA memcpy2D

***Matrix Copy***
- simply copy a matrix to another
  - C[i,j] = A[i,j]
  - `pitched matrices` for the best performance
  - for simplicity, we assume `square matrices`
- 다른 matrix 연산의 입장에서는?
  - theoretical limit 이론적 한계for matrix operations
  - the best score 최고 기록 for any matrix operations 행렬 대입 연산이 최고 기록이다.
  - 따라서 다른 연산, 즉 더하기 연산 등은 이 행렬 대입연산의 속도에 근접할 수록 최적화가 잘 된것이고
  - 이 기록을 추월했다면, 만세를 부를게 아니라, 뭔가 잘 못된 것임
    - 즉, 연산의 물체를 발견할 수 있는 어떤 기준이 될수도 있음

[Return Par4 Matrix Multiply](../README.md)  