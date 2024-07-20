cmake_minimum_required(VERSION 3.27)
project(giga_add_host_cpu CUDA)
project(giga_add_cuda_single_core CUDA)
project(giga_add_cuda_multi_core CUDA)
project(giga_add_clock CUDA)
project(giga_add_augmentation CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(giga_add_host_cpu
        giga_add_host_cpu.cu
        common.cpp
)

add_executable(giga_add_cuda_single_core
        giga_add_cuda_single_core.cu
        common.cpp
)

add_executable(giga_add_cuda_multi_core
        giga_add_cuda_multi_core.cu
        common.cpp
)

add_executable(giga_add_clock
        giga_add_clock.cu
        common.cpp
)

add_executable(giga_add_augmentation
        giga_add_augmentation.cu
        common.cpp
)


set_target_properties(
        giga_add_host_cpu
        giga_add_cuda_single_core
        giga_add_cuda_multi_core
        giga_add_clock
        giga_add_augmentation
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
