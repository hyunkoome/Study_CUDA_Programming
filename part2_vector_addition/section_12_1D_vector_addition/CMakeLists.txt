cmake_minimum_required(VERSION 3.27)
project(vec_add_host CUDA)
project(vec_add_host_kernel CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(vec_add_host
        vec_add_host.cu
        common.cpp
)

add_executable(vec_add_host_kernel
        vec_add_host_kernel.cu
        common.cpp
)

set_target_properties(
        vec_add_host
        vec_add_host_kernel
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
