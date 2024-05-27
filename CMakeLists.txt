cmake_minimum_required(VERSION 3.25)
project(AIComputeLabs_GPU CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(AIComputeLabs_GPU main.cu)

set_target_properties(AIComputeLabs_GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
