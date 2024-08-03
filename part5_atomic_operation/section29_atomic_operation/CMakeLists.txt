cmake_minimum_required(VERSION 3.27)
project(count_race CUDA)
project(count_atomic CUDA)
project(count_shared CUDA)
project(count_cas CUDA)
project(count_block LANGUAGES CUDA)

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

add_executable(count_race
        count_race.cu
        common.cpp
)

add_executable(count_atomic
        count_atomic.cu
        common.cpp
)

add_executable(count_shared
        count_shared.cu
        common.cpp
)

add_executable(count_cas
        count_cas.cu
        common.cpp
)

add_executable(count_block
        count_block.cu
        common.cpp
)

set_target_properties(
        count_race
        count_atomic
        count_shared
        count_cas
        count_block
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
