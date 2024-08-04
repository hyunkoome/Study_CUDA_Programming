cmake_minimum_required(VERSION 3.27)
project(qsort_block CUDA)
project(std_sort_block CUDA)
project(cpu_bubble_block CUDA)
project(cpu_even_odd_block.cu CUDA)
project(even_odd_block CUDA)
project(even_odd_global CUDA)
project(even_odd_outer_bubble CUDA)

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

add_executable(qsort_block
        qsort_block.cu
        common.cpp
)

add_executable(std_sort_block
        std_sort_block.cu
        common.cpp
)

add_executable(cpu_bubble_block
        cpu_bubble_block.cu
        common.cpp
)

add_executable(cpu_even_odd_block
        cpu_even_odd_block.cu
        common.cpp
)

add_executable(even_odd_block
        even_odd_block.cu
        common.cpp
)

add_executable(even_odd_global
        even_odd_block.cu
        common.cpp
)

add_executable(even_odd_outer_bubble
        even_odd_outer_bubble.cu
        common.cpp
)

set_target_properties(
        qsort_block
        std_sort_block
        cpu_bubble_block
        cpu_even_odd_block
        even_odd_block
        even_odd_global
        even_odd_outer_bubble
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
