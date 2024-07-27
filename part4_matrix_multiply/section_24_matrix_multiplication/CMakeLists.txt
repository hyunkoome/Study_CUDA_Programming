cmake_minimum_required(VERSION 3.27)
project(matmul_cpu_triple_for_loop CUDA)
project(matmul_cpu_outerK CUDA)
project(matmul_cuda_globalmem CUDA)
project(matmul_cuda_sharedmem_tiled_approach CUDA)
project(matmul_cuda_sharedmem_tiled_approach2 CUDA)
project(matmul_cuda_more_improve_alignedTile CUDA)
project(matmul_cuda_more_improve_aligned2 CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(matmul_cpu_triple_for_loop
        matmul_cpu_triple_for_loop.cu
        common.cpp
)

add_executable(matmul_cpu_outerK
        matmul_cpu_outerK.cu
        common.cpp
)

add_executable(matmul_cuda_globalmem
        matmul_cuda_globalmem.cu
        common.cpp
)

add_executable(matmul_cuda_sharedmem_tiled_approach
        matmul_cuda_sharedmem_tiled_approach.cu
        common.cpp
)

add_executable(matmul_cuda_sharedmem_tiled_approach2
        matmul_cuda_sharedmem_tiled_approach2.cu
        common.cpp
)

add_executable(matmul_cuda_more_improve_alignedTile
        matmul_cuda_more_improve_alignedTile.cu
        common.cpp
)

add_executable(matmul_cuda_more_improve_aligned2
        matmul_cuda_more_improve_aligned2.cu
        common.cpp
)

set_target_properties(
        matmul_cpu_triple_for_loop
        matmul_cpu_outerK
        matmul_cuda_globalmem
        matmul_cuda_sharedmem_tiled_approach
        matmul_cuda_sharedmem_tiled_approach2
        matmul_cuda_more_improve_alignedTile
        matmul_cuda_more_improve_aligned2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
