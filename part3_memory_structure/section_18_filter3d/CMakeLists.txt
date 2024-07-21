cmake_minimum_required(VERSION 3.27)
project(matrix3dfilter_host CUDA)
project(matrix3dfilter_cuda CUDA)
project(matrix3dfilter_3d_pitched_ptr CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(matrix3dfilter_host
        matrix3dfilter_host.cu
        common.cpp
)

add_executable(matrix3dfilter_cuda
        matrix3dfilter_cuda.cu
        common.cpp
)

add_executable(matrix3dfilter_3d_pitched_ptr
        matrix3dfilter_3d_pitched_ptr.cu
        common.cpp
)

set_target_properties(
        matrix3dfilter_host
        matrix3dfilter_cuda
        matrix3dfilter_3d_pitched_ptr
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
