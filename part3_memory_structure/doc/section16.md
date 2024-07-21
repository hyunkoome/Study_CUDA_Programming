# Memory Structure

### matrix addition: 행렬끼리 더하기 
- 1억개의 2D 행렬 더하기
  - cpu
    - 164005 usec = 0.164초
  - cuda
    - 1349 usec = 0.001349초 
  - cuda xy swap (x, y 역할 바꿈)
    - 3200 usec = 0.0032초
    - 속도가 느려짐
  - (결론)
    - 별거 아닌거 같은데, arry index를 헷갈려서 역할을 바꿔버리면, 
    - 간단한 더하기 할때도, 4배가 느려진다.
    - 그래서, `CUDA로 프로그래밍 할때는, 항상 array index 주의해라!`
      - `row 가 y 이고, column 이 x 라는 것을 주의 해서`
      - `y * width + x 로 계산 해야 한다!`


[Return Par3 Memory Structure](../README.md)  