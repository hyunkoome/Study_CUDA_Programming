cmake_minimum_required(VERSION 3.27)
project(cuda_error_check CUDA)
project(cuda_error_detect CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(cuda_error_check
        error_check.cu
        common.cpp
)

add_executable(cuda_error_detect
        error_detected.cu
)

set_target_properties(
        cuda_error_check
        cuda_error_detect
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
