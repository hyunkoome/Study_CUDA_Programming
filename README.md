# Study_CUDA_Programming

Must prepare as follows:
- Nvidia GPU
- OS: Ubuntu 20.04 (for me), Windows 10 over., Mac
- Install CUDA (for me, v12.1), download [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- Pure python Env (Not conda Env)
  - If you want to set python3 for main python module, please set.
```shell
sudo apt update
sudo apt install python3.8
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10 # If don't use python3, no need it
sudo update-alternatives --config python3 # If don't use python3, no need it
```
- Install glfw3 packages
```shell
sudo apt-get install libglfw3-dev libglfw3
```







