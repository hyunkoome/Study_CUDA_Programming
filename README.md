# Study_CUDA_Programming (based on C++ ver11)

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

- This tutorial is structured as follows:
  - `part1_cuda_kernel`: [Start CUDA programming](./part1_cuda_kernel/README.md)
    - print hello cuda (on Ubuntu) 
    - memory copy
    - add vector by using cpu or CUDA
    - error check
  - `part2_vector_addition`: [Study CUDA kernel launch](./part2_vector_addition/README.md)
    - elapsed time
    - CUDA kernel launch
    - 1d vector addition
    - Giga vector addition
    - AXPY and FMA
      - single precision
      - linear interpolation
    - thread and GPU
  - `part3_memory_structure`: [Memory Structure](./part3_memory_structure/README.md)
    - 메모리 계층 구조
    - CUDA 전용의 2D 메모리 할당 함수, pitched point 사용법 
    - 3D 행렬 사용 및 pitched point 사용법
    - CUDA 메모리 계층 구조
    - 인접 원소끼리 차이 구하기: shared memory 활용 
  - `part4_matrix_multiply`: [Matrix Multiply](./part4_matrix_multiply/README.md)
    - matrix copy
    - Matrix Transpose 전치 행렬
    - Matrix Multiplication
    - GEMM: general matrix-to-matrix multiplication
    - 메모리에 따른 CUDA 변수 스피드 측정
    - 정밀도와 속도개선
  - `part5_atommic_operation`: [Atomic Operation](./part5_atomic_operation/README.md)
    - Control Flow
      - if 문 과 for loop 문 어떻게 최적화 할것인지?
      - shared 메모리를 사용하는 경우라면, `half-by-half`를 사용하는 `even-odd` 보다 조금더 빠르다.!!
    - race conditions 문제의 해결방법으로 Atomic Operation 사용 
    - atomic operation 사용하여 histogram 구하기
    - Reduction Problem 솔루션 
    - GEMV operation
  
