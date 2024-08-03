cmake_minimum_required(VERSION 3.27)
project(gemv_host CUDA)
project(gemv_cuda_globalmem CUDA)
project(gemv_cuda_transpose CUDA)
project(gemv_cuda_tiled CUDA)
project(gemv_cuda_2d_kernel_partial CUDA)

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

add_executable(gemv_host
        gemv_host.cu
        common.cpp
)

add_executable(gemv_cuda_globalmem
        gemv_cuda_globalmem.cu
        common.cpp
)

add_executable(gemv_cuda_transpose
        gemv_cuda_transpose.cu
        common.cpp
)

add_executable(gemv_cuda_tiled
        gemv_cuda_tiled.cu
        common.cpp
)

add_executable(gemv_cuda_2d_kernel_partial
        gemv_cuda_2d_kernel_partial.cu
        common.cpp
)

set_target_properties(
        gemv_host
        gemv_cuda_globalmem
        gemv_cuda_transpose
        gemv_cuda_tiled
        gemv_cuda_2d_kernel_partial
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
