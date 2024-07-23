cmake_minimum_required(VERSION 3.27)
project(matcpy_cpu_for_loop CUDA)
project(matcpy_cpu_memcpy CUDA)
project(matcpy_cuda_pitched_approach_global_memory.cu CUDA)
project(matcpy_cuda_tiled_approach_shared_memory.cu CUDA)
project(matcpy_cuda_cudaMemcpy2D.cu CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(matcpy_cpu_for_loop
        matcpy_cpu_for_loop.cu
        common.cpp
)

add_executable(matcpy_cpu_memcpy
        matcpy_cpu_memcpy.cu
        common.cpp
)

add_executable(matcpy_cuda_pitched_approach_global_memory
        matcpy_cuda_pitched_approach_global_memory.cu
        common.cpp
)

add_executable(matcpy_cuda_tiled_approach_shared_memory
        matcpy_cuda_tiled_approach_shared_memory.cu
        common.cpp
)

add_executable(matcpy_cuda_cudaMemcpy2D
        matcpy_cuda_cudaMemcpy2D.cu
        common.cpp
)

set_target_properties(
        matcpy_cpu_for_loop
        matcpy_cpu_memcpy
        matcpy_cuda_pitched_approach_global_memory
        matcpy_cuda_tiled_approach_shared_memory
        matcpy_cuda_cudaMemcpy2D
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
