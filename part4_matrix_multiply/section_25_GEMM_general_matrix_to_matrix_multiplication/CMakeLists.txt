cmake_minimum_required(VERSION 3.27)
project(gemm_cpu CUDA)
project(gemm_cpu_outerK CUDA)
project(gemm_cuda_globalmem CUDA)
project(gemm_cuda_alignedTile CUDA)
project(gemm_cuda_tile CUDA)
project(gemm_cuda_tile_optim CUDA)


if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(gemm_cpu
        gemm_cpu.cu
        common.cpp
)

add_executable(gemm_cpu_outerK
        gemm_cpu_outerK.cu
        common.cpp
)

add_executable(gemm_cuda_globalmem
        gemm_cuda_globalmem.cu
        common.cpp
)

add_executable(gemm_cuda_alignedTile
        gemm_cuda_alignedTile.cu
        common.cpp
)

add_executable(gemm_cuda_tile
        gemm_cuda_tile.cu
        common.cpp
)

add_executable(gemm_cuda_tile_optim
        gemm_cuda_tile_optim.cu
        common.cpp
)


set_target_properties(
        gemm_cpu
        gemm_cpu_outerK
        gemm_cuda_globalmem
        gemm_cuda_alignedTile
        gemm_cuda_tile
        gemm_cuda_tile_optim
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
