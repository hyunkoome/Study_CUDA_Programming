# Study_CUDA_Programming (based on C++ ver11)
- All materials in this repository are based on lectures and code from Inflean's CUDA course. 
- The link of the course is as follows: 
  - https://www.inflearn.com/roadmaps/654#community
- All lecture materials and codes follow the instructor's license and are only for educational purposes for the course attended. 
- Commercial use is prohibited !!


### Must prepare as follows:
- Nvidia GPU
- OS: Ubuntu 20.04 (for me), Windows 10 over., Mac
- Install CUDA (for me, v12.1)
- Pure python Env (Not conda Env)
  - If you want to set python3 for main python module, please set.
```shell
sudo apt update
sudo apt install python3.8
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10 # If don't use python3, no need it
sudo update-alternatives --config python3 # If don't use python3, no need it
```

### Install glfw3 packages
```shell
sudo apt-get install libglfw3-dev libglfw3
```

### Install cmake 3.30
```shell
sudo apt purge cmake
sudo apt install wget build-essential

wget https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0.tar.gz
tar -xvzf cmake-3.30.0.tar.gz
cd cmake-3.30.0
./bootstrap --prefix=/usr/local
make
sudo make install
cmake --version
```
- If you don't find cmake version, please edit as follows:
```shell
vi ~./bashrc
```
PATH=/usr/local/bin:$PATH:$HOME/bin
```shell
source ~./bashrc
```
**Download [CUDA Samples](https://github.com/NVIDIA/cuda-samples) (for me, v12.1)**
```shell
wget https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v12.1.tar.gz
tar -zxvf v12.1.tar.gz
```

```shell
wget https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v12.1.zip
unzip v12.1.zip
make
sudo make install
```

### CUDA for Ubuntu

- $ ubuntu-drivers devices
- $ sudo apt install nvidia-driver-xx
  - reboot !
- $ nvidia-smi (only for checking your NVIDIA driver)
  - visit CUDA-zone to get the CUDA toolkit
- $ sudo apt get install build-essential (to get GCC compilers)
- $ nvcc -V (now you should get the NVIDIA CUDA Compiler messages)

### CUDA Tutorial
- in each section, build the project as shown below and run the generated file.
```shell
mkdir build
cd build
cmake ..
make
./generated_execution_file
```

### This tutorial is structured as follows:

#### 1. `part1_cuda_kernel`: [Start CUDA programming](./part1_cuda_kernel/README.md) | [Certificate](https://www.inflearn.com/certificate/962884-329543-12987047)
- print hello cuda (on Ubuntu) 
- memory copy
- add vector by using cpu or CUDA
- error check

#### 2. `part2_vector_addition`: [Study CUDA kernel launch](./part2_vector_addition/README.md) | [Certificate](https://www.inflearn.com/certificate/962884-329544-12987046)
- elapsed time
- CUDA kernel launch
- 1d vector addition
- Giga vector addition
- AXPY and FMA
  - single precision
  - linear interpolation
- thread and GPU

#### 3. `part3_memory_structure`: [Memory Structure](./part3_memory_structure/README.md) | [Certificate](https://www.inflearn.com/certificate/962884-329600-12987048)
- 메모리 계층 구조
- CUDA 전용의 2D 메모리 할당 함수, pitched point 사용법 
- 3D 행렬 사용 및 pitched point 사용법
- CUDA 메모리 계층 구조
- 인접 원소끼리 차이 구하기: shared memory 활용

#### 4. `part4_matrix_multiply`: [Matrix Multiply](./part4_matrix_multiply/README.md) | [Certificate](https://www.inflearn.com/certificate/962884-329601-12987045)
- matrix copy
- Matrix Transpose 전치 행렬
- Matrix Multiplication
- GEMM: general matrix-to-matrix multiplication
- 메모리에 따른 CUDA 변수 스피드 측정
- 정밀도와 속도개선

#### 5. `part5_atommic_operation`: [Atomic Operation](./part5_atomic_operation/README.md) | [Certificate](https://www.inflearn.com/certificate/962884-329721-12987044)
- Control Flow
  - if 문 과 for loop 문 어떻게 최적화 할것인지?
  - shared 메모리를 사용하는 경우라면, `half-by-half`를 사용하는 `even-odd` 보다 조금더 빠르다.!!
- race conditions 문제의 해결방법으로 Atomic Operation 사용 
- atomic operation 사용하여 histogram 구하기
- Reduction Problem 솔루션 
- GEMV operation

#### 6. `part6_search_sort`: [Search & Sort](./part6_search_sort/README.md) | [Certificate](https://www.inflearn.com/certificate/962884-329723-12987043)
- Linear Search 선형 탐색
- Search All 모든 위치 모두 찾기
  - CUDA에서 stride 사용하는 것이 제일 빠르다. 
- Binary Search 이진 탐색
  - CUDA 사용해서, `binary search`는 효과적이지 못하다.
  - `그냥 CPU 사용하세요!. 특히 STL 짱짱 빠름.`
- `CUDA 에서 Sort 하는 방법`.. 본격적으로 얘기해 보자!!
  - 블럭 단위 parallel sorting
    - CUDA even-odd sort: 엄청 빨라 짐
  - global 메모리 활용 parallel sort 할때는,
    - CUDA (even-odd) 에서 도차도 상당히 느리다.
- Bitonic Sort 바이토닉 소트
  - 병렬 처리를 위한, 소팅 방법이라고 보면 됨
- Counting Merge Sort 카운팅 방식 머지 소트 (병합 정렬)
  - 병렬 처리에 가장 적합한 Large Scale Parallel Counting Merge Sort 방법


## Additional Comments
- All description in the materials have been modified by myself, Hyunkoo Kim.
- (c) 2024. hyunkookim.me@gmail.com. All rights reserved. 
