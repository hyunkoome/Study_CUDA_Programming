cmake_minimum_required(VERSION 3.27)
project(cpu_vector_add CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()

set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(cpu_vector_add cpu_vector_add.cu)

set_target_properties(
        cpu_vector_add
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
