cmake_minimum_required(VERSION 3.27)
project(saxpy_cpu_host CUDA)
project(saxpy_cuda CUDA)
project(saxpy_cuda_fma CUDA)
project(lerp_cuda CUDA)
project(lerp_cuda_fma CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(saxpy_cpu_host
        saxpy_cpu_host.cu
        common.cpp
)

add_executable(saxpy_cuda
        saxpy_cuda.cu
        common.cpp
)

add_executable(saxpy_cuda_fma
        saxpy_cuda_fma.cu
        common.cpp
)

add_executable(lerp_cuda
        lerp_cuda.cu
        common.cpp
)

add_executable(lerp_cuda_fma
        lerp_cuda_fma.cu
        common.cpp
)

set_target_properties(
        saxpy_cpu_host
        saxpy_cuda
        saxpy_cuda_fma
        lerp_cuda
        lerp_cuda_fma
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
