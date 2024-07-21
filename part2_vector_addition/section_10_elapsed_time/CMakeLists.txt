cmake_minimum_required(VERSION 3.27)
project(chrono CUDA)
project(chrono_macro CUDA)
project(chronoClock CUDA)
project(sleep CUDA)
project(argc CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 11)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(chrono
        chrono.cu
)

add_executable(chrono_macro
        chrono_macro.cu
        common.cpp
)

add_executable(chronoClock
        chronoClock.cu
)

add_executable(sleep
        sleep.cu
)

add_executable(argc
        argc.cu
)

set_target_properties(
        chrono
        chrono_macro
        chronoClock
        sleep
        argc
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
