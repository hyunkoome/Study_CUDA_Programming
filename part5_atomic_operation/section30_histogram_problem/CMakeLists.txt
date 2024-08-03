cmake_minimum_required(VERSION 3.27)
project(hist_host CUDA)
project(hist_host_4bytes CUDA)
project(hist_cuda_globalmem CUDA)
project(hist_cuda_globalmem_4bytes CUDA)
project(hist_cuda_sharedmem CUDA)
project(hist_cuda_sharedmem_4bytes CUDA)

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

add_executable(hist_host
        hist_host.cu
        common.cpp
        image.cpp
)

add_executable(hist_host_4bytes
        hist_host_4bytes.cu
        common.cpp
        image.cpp
)

add_executable(hist_cuda_globalmem
        hist_cuda_globalmem.cu
        common.cpp
        image.cpp
)

add_executable(hist_cuda_globalmem_4bytes
        hist_cuda_globalmem_4bytes.cu
        common.cpp
        image.cpp
)

add_executable(hist_cuda_sharedmem
        hist_cuda_sharedmem.cu
        common.cpp
        image.cpp
)

add_executable(hist_cuda_sharedmem_4bytes
        hist_cuda_sharedmem_4bytes.cu
        common.cpp
)

set_target_properties(
        hist_host
        hist_host_4bytes
        hist_cuda_globalmem
        hist_cuda_globalmem_4bytes
        hist_cuda_sharedmem
        hist_cuda_sharedmem_4bytes
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
