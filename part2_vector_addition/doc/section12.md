# Vector addition

### 실제 예제활용 CPU vs CUDA

***`vec_add_cuda_single_core.cu`***
- single core로 1개의 core만 사용하는 걸로..
- 아마도, gpu core 1개는 cpu보다 느리니, cpu core 사용한거보다 느리게 나올듯.. 확인해보자!!
- 확인해보면 cpu 1개로 돌리는 것 보다, CUDA 1개로 돌리는게 엄청~ 느려짐

***`vec_add_cuda_error.cu`***
- SIZE  = 100만개(1024 x 1024) core를 사용 요구 해서 돌림
- error 뜨는 이유: 우리 GPU에 100만게 core가 없음.
- 그런데, 100만개의 코어를 요구해서 돌리려고 함 => 안되죠 !
- 그래서, 안되는 것을 한번, 확인해보자!
- 에러 메세지:
  - cuda failure "invalid configuration argument" : CUDA 커널을 만들때 이 커널이 잘못 configure 됬다는 의미임
- M (streaming multi-processor)에서 1M개의 thread를 동시 실행 불가능 -> 실제로는 1024개가 한계

***`vec_add_cuda_1024_core.cu`***
- vector 크기 SIZE = 1024x1024, 즉, 100만개
- blockDim = 1024: 쓰레드 블록에는 1024개를 돌리도록 하고, 즉, 블록 1개 안에는 1024개가 돌아가고
- gridDim = SIZE / 1024: 블록 개수 계산, 100만개를 이 쓰레드 블럭 1개에 1024개가 돌라가니깐, 1M/1024 나눠버리면 => 블록의 개수가 나옴
- 이제, cpu 보다 빨리 돌아가는 것을 확인 가능,
- SIZE = 1024 * 1024 * 32 , 즉 32M 만개로 세팅해도 잘 작동. cpu 성능 차이 확연히 확인 가능

***`vec_add_class.cu`***
- CUDA 커널 함수는 kernel function
  - c++ template 적용은 가능
  - `class member 로는 불가능 !!`
  - 그래서, 커널 함수는 외부의 external void 함수로 선언하는 게 일반적임!!
    - `__global__ void` kernelVecAdd(){...}
- C 코드 보다는, C++ 로 바꾸면, 코드는 동작하지만,
  - C++ 바꾸면 클래스 핸들링하고 하는데, 시간이 조금 더 걸릴 수 있음
  - 원하면 C++ 바꾸면 되지만,
  - `확장성 등 고려해서, 코드는 가능하면 C 형태로 사용하는 것이 좋다!`


[Return Par2 Vector Addition](../README.md)  