cmake_minimum_required(VERSION 3.27)
project(matadd_host_cpu CUDA)
project(matadd_cuda CUDA)
project(matadd_xyswap CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(matadd_host_cpu
        matadd_host_cpu.cu
        common.cpp
)

add_executable(matadd_cuda
        matadd_cuda.cu
        common.cpp
)

add_executable(matadd_xyswap
        matadd_xyswap.cu
        common.cpp
)

set_target_properties(
        matadd_host_cpu
        matadd_cuda
        matadd_xyswap
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
