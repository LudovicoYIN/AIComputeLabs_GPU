cmake_minimum_required(VERSION 3.25)
project(AIComputeLabs_GPU CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(AIComputeLabs_GPU 1_hello_cuda.cu 2_indexing.cu)
add_executable(1_hello_cuda 1_hello_cuda.cu)
add_executable(2_indexing 2_indexing.cu)
add_executable(3_vector_add 3_vector_add.cu)
add_executable(4_device_query 4_device_query.cu)

set_target_properties(AIComputeLabs_GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
