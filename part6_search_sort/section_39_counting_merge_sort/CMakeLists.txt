cmake_minimum_required(VERSION 3.27)
project(cpu_merge_block CUDA)
project(merge_increasing_block CUDA)
project(merge_block CUDA)
project(merge_outer_bubble CUDA)
project(merge_outer_bubble_large CUDA)
project(merge_global CUDA)
project(merge_global_large CUDA)

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

add_executable(cpu_merge_block
        cpu_merge_block.cu
        common.cpp
)

add_executable(merge_increasing_block
        merge_increasing_block.cu
        common.cpp
)

add_executable(merge_block
        merge_block.cu
        common.cpp
)

add_executable(merge_outer_bubble
        merge_outer_bubble.cu
        common.cpp
)

add_executable(merge_outer_bubble_large
        merge_outer_bubble_large.cu
        common.cpp
)

add_executable(merge_global
        merge_global.cu
        common.cpp
)

add_executable(merge_global_large
        merge_global_large.cu
        common.cpp
)

set_target_properties(
        cpu_merge_block
        merge_increasing_block
        merge_block
        merge_outer_bubble
        merge_outer_bubble_large
        merge_global
        merge_global_large
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
