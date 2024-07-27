cmake_minimum_required(VERSION 3.27)
project(speed_variables CUDA)
project(mem_clear_cpu CUDA)
project(mem_clear_cuda CUDA)


if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(speed_variables
        speed_variables.cu
        common.cpp
)

add_executable(mem_clear_cpu
        mem_clear_cpu.cu
        common.cpp
)

add_executable(mem_clear_cuda
        mem_clear_cuda.cu
        common.cpp
)


set_target_properties(
        speed_variables
        mem_clear_cpu
        mem_clear_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
