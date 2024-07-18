cmake_minimum_required(VERSION 3.27)
project(memcpy CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()

set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(memcpy memcpy.cu)

set_target_properties(
        memcpy
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
