cmake_minimum_required(VERSION 3.27)
project(saxpy_cuda_fma CUDA)
project(saxpy_symbol CUDA)
project(saxpy_addr CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(saxpy_cuda_fma
        saxpy_cuda_fma.cu
        common.cpp
)

add_executable(saxpy_symbol
        saxpy_symbol.cu
        common.cpp
)

add_executable(saxpy_addr
        saxpy_addr.cu
        common.cpp
)

set_target_properties(
        saxpy_cuda_fma
        saxpy_symbol
        saxpy_addr
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
