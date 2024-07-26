cmake_minimum_required(VERSION 3.27)
project(transpose_cpu CUDA)
project(transpose_cuda_globalmem_naive CUDA)
project(transpose_cuda_sharedmem_naive CUDA)
project(transpose_cuda_sharedmem_optimized CUDA)
project(transpose_cuda_sharedmem_optimized_backconflict CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(transpose_cpu
        transpose_cpu.cu
        common.cpp
)

add_executable(transpose_cuda_globalmem_naive
        transpose_cuda_globalmem_naive.cu
        common.cpp
)

add_executable(transpose_cuda_sharedmem_naive
        transpose_cuda_sharedmem_naive.cu
        common.cpp
)

add_executable(transpose_cuda_sharedmem_optimized
        transpose_cuda_sharedmem_optimized.cu
        common.cpp
)

add_executable(transpose_cuda_sharedmem_optimized_backconflict
        transpose_cuda_sharedmem_optimized_backconflict.cu
        common.cpp
)

set_target_properties(
        transpose_cpu
        transpose_cuda_globalmem_naive
        transpose_cuda_sharedmem_naive
        transpose_cuda_sharedmem_optimized
        transpose_cuda_sharedmem_optimized_backconflict
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
