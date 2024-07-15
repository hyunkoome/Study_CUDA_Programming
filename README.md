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

**header 관련해서**
- nvcc compiler 사용시 아래 3개 header는 자동으로 적용되는 경우가 많지만, 추가해주도록 !!
```c++
#include <cuda.h> // bisic header file
#include <cuda_runtime_api.h> // compiler 사용시, runtime function을 사용가능, API = application programming interface (함수 정의)
#include <cuda_runtime.h> // CUDA built-ins, types 등 사용 가능
```
- not CUDA standard header file, provided by CUDA samples,  you can find them in CUDA samples / inc directory
```c++
#include <helper_cuda.h>
#include <helper_functions.h>
```
- 모든 CUDA 함수는 cuda로 시작
- 대부분 모든 CUDA 함수는 에러 코드를 리턴, (성공 시는 cudaSuccess 를 리턴)
- Please check [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
  - nvcc: -gencode 
    - GeForce RTX 4090: Compute Capability - 8.9
    - –gencode=arch=compute_89,code=\"sm_89,compute_89\"
      –arch=sm_89
  - for more details, read the NVCC manual
  - set it in your “makefile” (or other equivalents)


**Easy way to compile**
```shell
nvcc -o hello_cuda hello_cuda.cu -DCMAKE_CUDA_ARCHITECTURES='89'
```
- here: -O2 (오투) : build release mode
```shell
nvcc --gpu-architecture=compute_89 --gpu-code=sm_89,compute_89 -O2 -o hello_linux hello_linux.cu
```
- add architecture
```shell
/usr/local/cuda/bin/nvcc -ccbin g++ -I../../../Common  -m64 --std=c++14 --threads 1 -gencode arch=compute_89,code=sm_89 -o segmentationTree.o -c segmentationTree.cu
```

**build**
- In each folder
```shell
mkdir build
cd build
cmake ..
make
```



