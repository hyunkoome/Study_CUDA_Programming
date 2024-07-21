cmake_minimum_required(VERSION 3.27)
project(adjdiff_host CUDA)
project(adjdiff_cuda CUDA)
project(adjdiff_shared_memory CUDA)
project(adjdiff_overuse CUDA)
project(adjdiff_shared2 CUDA)
project(adjdiff_devQuery CUDA)

project(shared_mem CUDA)
project(race_cond CUDA)
project(synced CUDA)
project(ptr_test CUDA)
project(param_test CUDA)

if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda-12.1/include")
endif ()
include_directories("./")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89") # RTX4090

add_executable(adjdiff_host
        adjdiff_host.cu
        common.cpp
)

add_executable(adjdiff_cuda
        adjdiff_cuda.cu
        common.cpp
)

add_executable(adjdiff_shared_memory
        adjdiff_shared_memory.cu
        common.cpp
)

add_executable(adjdiff_overuse
        adjdiff_overuse.cu
        common.cpp
)

add_executable(adjdiff_shared2
        adjdiff_shared2.cu
        common.cpp
)

add_executable(adjdiff_devQuery
        adjdiff_devQuery.cu
        common.cpp
)

add_executable(shared_mem
        shared_mem.cu
        common.cpp
)

add_executable(race_cond
        race_cond.cu
        common.cpp
)

add_executable(synced
        synced.cu
        common.cpp
)

add_executable(ptr_test
        ptr_test.cu
        common.cpp
)

add_executable(param_test
        param_test.cu
        common.cpp
)

set_target_properties(
        adjdiff_host
        adjdiff_cuda
        adjdiff_shared_memory
        adjdiff_overuse
        adjdiff_shared2
        adjdiff_devQuery
        shared_mem
        race_cond
        synced
        ptr_test
        param_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
