cmake_minimum_required(VERSION 3.27)
project(matadd_cuda CUDA)
project(matadd_pitched CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(matadd_cuda
        matadd_cuda.cu
        common.cpp
)

add_executable(matadd_pitched
        matadd_pitched.cu
        common.cpp
)

set_target_properties(
        matadd_cuda
        matadd_pitched
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
