cmake_minimum_required(VERSION 3.27)
project(search_all CUDA)
project(search_all_find CUDA)
project(search_all_atomic CUDA)
project(search_all_block CUDA)
project(search_all_stride CUDA)
project(search_all_stride2 CUDA)

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

add_executable(search_all
        search_all.cu
        common.cpp
)

add_executable(search_all_find
        search_all_find.cu
        common.cpp
)

add_executable(search_all_atomic
        search_all_atomic.cu
        common.cpp
)

add_executable(search_all_block
        search_all_block.cu
        common.cpp
)

add_executable(search_all_stride
        search_all_stride.cu
        common.cpp
)

add_executable(search_all_stride2
        search_all_stride2.cu
        common.cpp
)

set_target_properties(
        search_all
        search_all_find
        search_all_atomic
        search_all_block
        search_all_stride
        search_all_stride2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
