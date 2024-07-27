# Memory Structure

***floating point numbers 실수 표현 방법***
- IEEE 754 floating-point standard `부동 소수점 방식` 
  - `float`, `double`, `long double`
  - real number = sign 부호 x 크기 x 2^지수
    - 1비트: sign 부호(0이면 양수, 1이면 음수)
    - 8비트: 지수 부분(정수로 표현, 음수도 표현, 값에서..-128 빼서.. 이것이 2의 거듭제곱 형태가 됨)
    - 23비트: 크기 부분
      - 이진수 1.00000 < 크기(1.xxxx) < 이진수 10.000000
      - 1은 생략 -> 1비트 아낌.
  - 10 진수에서는 과학적 표기법: ±1.234 × 10 ±56

***정밀도 문제***
- algorithmic consideration
  - Floating-point operations are NOT strictly associative
  - `(small + small) + large → may be more accurate`
  - (large + small) + small → pot
  - 아주 작은 숫자를 큰숫자에 더하면, 아주 작은 숫자가 사라짐 
    - 왜냐.. `소수점 7자리까지 밖에 저장`을 못하므로.
    - 그래서, `아주 작은 숫자들 끼리 먼저 더해서.. 큰 숫자에 더하는 것이..`
    - `계산 오차를 줄여줌!!` 
- `common.cpp` 의 `getNaiveSum()`과 `getSUM()` 함수 확인!!
  - `getNaiveSum()`: 아주 작은 숫자들 .. 큰숫자에 더해지니.. 더해지지않고.. 
    - 계속 소멸되는 버그..있음
  - `getSUM()`: `partial sum` 구하는 방법으로 변경하여, `오차 개선`!

***float-safe 최적화***
- `float x = 1.0f;` 
  - `1` 로 적으면, int로  잡혀서, 형 변환이 필요하여, 추가 clock 필요!
  - 그래서, `f`붙이는 습관!!!
- float y = 1.0f * x;
  - `1.0`이면 double로 잡혀서, 형 변환이 필요하여, 추가 clock 필요!
  - 그래서, `f`붙이는 습관!!!
- float z = 3.0f * sinf( y );
  - 일반 수학 함수는 double 형을 위한 것으로..
  - `float 형 을 위한 함수는 보통 뒤에. f가 붙음` 
    - 속도가 조금이라도 빨라짐 

***CUDA Runtime MATH library***
- There are `two types` of runtime math operations
- `func( )` : compile to ***multiple instructions*** (or library functions)
  - `slower but higher accuracy` 정밀도가 목적..
  - examples: sin(x), sinf(x), exp(x), expf(x), pow(x, y), powf(x, y)
- `__func( )` : direct mapping to a ***single hardware instruction***
  - `fast but low accuracy` 속도가 목적..
  - examples: __sin(x), __sinf(x), __exp(x), __expf(x), __pow(x, y)
- `–use_fast_math` ***compiler option***:
  - forces every `func( )` to compile to `__func( )`
  - 컴파일 시, nvcc 에, `–use_fast_math` 를 추가하면, 우리가 사용한 모든 수학 함수를 __가 붙은 함수로 바꿔서 컴파일 시켜버림!!
    - 디버그 시에는 일반적인 함수를 사용해서 개발하다가, 
    - release 시에는 이 옵션을 붙여서.. 컴파일 하면,,
    - 소스코드를 수정하지 않고도.. 속도가 엄청 빨라짐..
    - 그러나, 정밀도가 떨어질 수도 있으니.. 테스트 해가면서..하는게 제일 안전 함!

***속도 비교***
- `sine / cosine` calculation (with a `single core`, not many-core)
  - CPU version: 16,915 usec    
  - CUDA default: 134,234 usec
    - 1개의 core만 썼으므로, cpu 보다 느릴수 밖에 없음.
  - CUDA fast-math (빠르지만, 근사값): 10,109 usec
    - 근사값이라고 하지만, 크게 오차가 나오지 않음.
    - 그래서, 많이 쓰자!!
  - CUDA sincosf( ): 11,873 usec
    - fast-math 보다 더 빨라진다고 하는데.. 실제로는 안빨라지는 듯함.
    - 근데, 강의에서는 빨라지는 듯.. 
    - 그래서, 나중에 사용할때,,확인이 필요할 것 같음
  - CUDA fma( ): 10,396 usec
    - fma: fused multiply-add instruction (z <- a x + y 문제)

[Return Par4 Matrix Multiply](../README.md)  