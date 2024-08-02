cmake_minimum_required(VERSION 3.27)
project(evenodd CUDA)
project(halfhalf CUDA)
project(evenodd_shared CUDA)
project(halfhalf_shared CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(evenodd
        evenodd.cu
        common.cpp
)

add_executable(halfhalf
        halfhalf.cu
        common.cpp
)

add_executable(evenodd_shared
        evenodd_shared.cu
        common.cpp
)

add_executable(halfhalf_shared
        halfhalf_shared.cu
        common.cpp
)

set_target_properties(
        evenodd
        halfhalf
        evenodd_shared
        halfhalf_shared
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
