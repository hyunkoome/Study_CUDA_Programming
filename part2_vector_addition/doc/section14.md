# Vector addition

### AXPY and FMA

- AXPY : a X plus Y problem
  - BLAS (Basic Linear Algebra Sybprograms)
    - 선형 대수, 행렬/벡터 계산에서 매우 유명한 표준 라이브러리
    - `CUDA 도 BLAS 함수를 제공하지만, 여기서는 직접 구현하겠다.`
  - Z = a X + Y
    - X, Y, Z 는 vector (1d array)
    - a 는 scalar (상수) 값
  - 선형 대수
  - 행렬을 계산하는데.. 가장 기본
- FMA : `fused multiply-add` instruction
  - 어셈블리 혹은 CPU나 GPU 레벨에서 multiply-add 를, 
  - 즉, 곱한 다음에 바로 더하는 것을 빠르게, 구현해 놓은 인스트럭션(명령어) 인데,
  - 이것을 사용하면 조금 더 빨라지긴 함

***AXPY 루틴***
  - SAXPY: AXPY 로직을 `single precision` 으로 만든 것
    - SAXPY: single precision (float)
    - DAXPY: double precision (double)
    - CAXPY: complex numbers 복소수 
  - SAXPY - CPU 구현
    - a: 1.234000
    - SIZE: 256개 vectors
    - 실행 시간: 413,729 usec = 0.413초
  - SAXPY - CUDA 구현
    - 실행 시간: 4,852 usec = 0.0048초

***FMA instruction (CUDA)***
- SAXPY - FMA 적용
  - FMA/FMAC: fused multiply-add / fused multiply-accumulate instruction
    - FMA == FMAC == FMADD
  - fused = blended = integrated
  - round(a * x + y)
  - 약간 빠름
  - `정밀도`가 약간 올라감 ! -> deep learning 등에서 이쪽을 선호
  - `in one step` !!
  - CPU / GPU 에서 machine instruction 으로 구현해서 제공
- in CUDA math library,
  - `float fmaf( float a, float x, float y );`
    - C/C++에서 함수 뒤에 `f` 가 붙으면 float를 위한 특별한 버전이다 라는 의미 임
    - ※ 평범한 함수 이름은 double에 관한 것임 
  - `double fma( double a, double x, double y );`
  - returns `(a * x + y)` with `FMA instruction`
- 실행 시간: 3,631 usec = 0.0036초
  - 그렇게 많이 빨라지지는 않았음
  - 왜? 
    - 나중에 복잡한 프로그램 짤때는 FMA 가 의미가 있을수 있는데, 지금은 너무 간단한 코드여서, 그리 차이는 없을수 있음
    - 우리가 쓴 컴파일러가, 현대의 C, C++, CUDA 컴파일러들은 우리가 생각했던 것 보다 훨씬 머리가 좋습니다. 
      - 내부적으로 최적화되서, 캄파일러가 내부적으로 fma를 사용했을수도 있음
- FMA를 적용하면 연산이 빨라지는 예시
  - 벡터 내적(dot product): `벡터 차원 개수의 fma 연산으로 가능` 
    - for two vectors (a x , a y , a z ) and (b x , b y , b z )
    - calculate (a x * b x) + (a y * b y) + (a z * b z)
    - FMA 계산: 곱하기 6번 + 더하기 2번의 연산이 fma 3번으로 바꾸면 속도가 빨라짐 
      - answer = 0
      - answer = fma( ax, bx , answer)
      - answer = fma( ay, by , answer)
      - answer = fma( az, bz , answer)
  - lerp러프: **l**inear int**erp**olation 선형 보간법, 가중 평균
    - f(t) = (1 – t) *  v0 + t * v1
      - v0 = (x0 , y0 )
      - v1 = (x1 , y1 )
    - FMA 계산: `fma 연산 2번으로 가능`
      - f(t) = (1 – t) * v0 + t * v1 = (v0 – t * v0 ) + t * v1
      - fma(t, v1 , fma( –t, v0 , v0))  

***LERP: linear interpolation***
- 프로그램 실행시
  - 기본 SIZE 로 동작: `./프로그램`  
  - SIZE만 바꾸고 싶으면: `./프로그램 SIZE`
    - SIZE에 k, K: kilo (= 1,024)로 해석
    - SIZE에 m, M: million (= 1,024 * 1,024)로 해석
  - SIZE와 a값도 바꾸고 싶으면: `./프로그램 SIZE a값`
- LERP - CUDA 구현
  - SIZE: 512,000,000
  - 실행 시간: 6770 usec = 0.006770 초
- LERP - FMA 적용
  - SIZE: 512,000,000
  - 실행 시간: 6734 usec = 0.006734 초


[Return Par2 Vector Addition](../README.md)  