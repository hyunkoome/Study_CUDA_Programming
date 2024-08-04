cmake_minimum_required(VERSION 3.27)
project(bsearch_host CUDA)
project(bsearch_host_bsearch CUDA)
project(bsearch_host_stl CUDA)
project(bsearch_cuda_block CUDA)

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

add_executable(bsearch_host
        bsearch_host.cu
        common.cpp
)

add_executable(bsearch_host_bsearch
        bsearch_host_bsearch.cu
        common.cpp
)

add_executable(bsearch_host_stl
        bsearch_host_stl.cu
        common.cpp
)

add_executable(bsearch_cuda_block
        bsearch_cuda_block.cu
        common.cpp
)

set_target_properties(
        bsearch_host
        bsearch_host_bsearch
        bsearch_host_stl
        bsearch_cuda_block
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
