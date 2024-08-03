cmake_minimum_required(VERSION 3.27)
project(sum_host CUDA)
project(sum_atomic CUDA)
project(sum_shared CUDA)
project(sum_reduce CUDA)
project(sum_reversed CUDA)
project(sum_addFirst CUDA)
project(sum_lastwarp CUDA)
project(sum_warpshuffle CUDA)
project(sum_twostep CUDA)

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

add_executable(sum_host
        sum_host.cu
        common.cpp
)

add_executable(sum_atomic
        sum_atomic.cu
        common.cpp
)

add_executable(sum_shared
        sum_shared.cu
        common.cpp
)

add_executable(sum_reduce
        sum_reduce.cu
        common.cpp
)

add_executable(sum_reversed
        sum_reversed.cu
        common.cpp
)

add_executable(sum_addFirst
        sum_addFirst.cu
        common.cpp
)

add_executable(sum_lastwarp
        sum_lastwarp.cu
        common.cpp
)

add_executable(sum_warpshuffle
        sum_reduce.cu
        common.cpp
)

add_executable(sum_twostep
        sum_twostep.cu
        common.cpp
)


set_target_properties(
        sum_host
        sum_atomic
        sum_shared
        sum_reduce
        sum_reversed
        sum_addFirst
        sum_lastwarp
        sum_warpshuffle
        sum_twostep
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
