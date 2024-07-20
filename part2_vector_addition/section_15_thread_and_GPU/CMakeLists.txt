cmake_minimum_required(VERSION 3.27)
project(warp_lane CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

#set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(saxpy_cpu_host
        warp_lane.cu
        common.cpp
)

set_target_properties(warp_lane PROPERTIES
        CXX_STANDARD 11
        CUDA_STANDARD 14
        CXX_STANDARD_REQUIRED ON
#        CUDA_ARCHITECTURES 89
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
