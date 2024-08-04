cmake_minimum_required(VERSION 3.27)
project(search_host_naive CUDA)
project(search_host_find CUDA)
project(search_cuda_atomic CUDA)
project(search_cuda_block CUDA)
project(search_cuda_stride CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 89)  # This is for CUDA 11.2 and newer
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89 -arch sm_89") # RTX4090

# Add the -w flag to suppress warnings
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w")

add_executable(search_host_naive
        search_host_naive.cu
        common.cpp
)

add_executable(search_host_find
        search_host_find.cu
        common.cpp
)

add_executable(search_cuda_atomic
        search_cuda_atomic.cu
        common.cpp
)

add_executable(search_cuda_block
        search_cuda_block.cu
        common.cpp
)

add_executable(search_cuda_stride
        search_cuda_stride.cu
        common.cpp
)

set_target_properties(
        search_host_naive
        search_host_find
        search_cuda_atomic
        search_cuda_block
        search_cuda_stride
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
