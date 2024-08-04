cmake_minimum_required(VERSION 3.27)
project(cpu_bitonic_block CUDA)
project(bitonic_block CUDA)
project(cpu_bitonic2_block CUDA)
project(bitonic2_block CUDA)
project(bitonic_outer_bubble CUDA)
project(bitonic_outer_bubble_large CUDA)

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

add_executable(cpu_bitonic_block
        cpu_bitonic_block.cu
        common.cpp
)

add_executable(bitonic_block
        bitonic_block.cu
        common.cpp
)

add_executable(cpu_bitonic2_block
        cpu_bitonic2_block.cu
        common.cpp
)

add_executable(bitonic2_block
        bitonic2_block.cu
        common.cpp
)

add_executable(bitonic_outer_bubble
        bitonic_outer_bubble.cu
        common.cpp
)

add_executable(bitonic_outer_bubble_large
        bitonic_outer_bubble_large.cu
        common.cpp
)

set_target_properties(
        cpu_bitonic_block
        bitonic_block
        cpu_bitonic2_block
        bitonic2_block
        bitonic_outer_bubble
        bitonic_outer_bubble_large
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
