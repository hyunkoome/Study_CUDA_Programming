cmake_minimum_required(VERSION 3.27)
project(vec_add_host CUDA)
project(vec_add_host_kernel CUDA)
project(vec_add_cuda_single_core CUDA)
project(vec_add_cuda_error CUDA)
project(vec_add_cuda_1024_core CUDA)
project(vec_add_class CUDA)

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

add_executable(vec_add_cuda_single_core
        vec_add_cuda_single_core.cu
        common.cpp
)

add_executable(vec_add_cuda_error
        vec_add_cuda_error.cu
        common.cpp
)

add_executable(vec_add_cuda_1024_core
        vec_add_cuda_1024_core.cu
        common.cpp
)

add_executable(vec_add_class
        vec_add_class.cu
        common.cpp
)

set_target_properties(
        vec_add_host
        vec_add_host_kernel
        vec_add_cuda_single_core
        vec_add_cuda_error
        vec_add_cuda_1024_core
        vec_add_class
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
