cmake_minimum_required(VERSION 3.27)
project(hello_cuda CUDA)

#set(DTCNN_CUDA_ARCHITECTURES 89)
#if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
#    set(CMAKE_CUDA_ARCHITECTURES 89)
#endif()

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(hello_cuda hello_cuda.cu)
add_executable(hello_parallel hello_parallel.cu)

set_target_properties(hello_cuda hello_parallel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
