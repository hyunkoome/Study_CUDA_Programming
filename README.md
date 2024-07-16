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
- `part1_cuda_kernel`: [Start CUDA programming](./part1_cuda_kernel/README.md)
  - print hello cuda (on Ubuntu) 
  - memory copy
  - add vector by using cpu or CUDA
  - error check
- 
  


