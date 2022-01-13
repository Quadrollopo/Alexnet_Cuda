cmake_minimum_required(VERSION 3.10)
project(Alexnet CUDA)

set(CMAKE_CUDA_STANDARD 14)
add_executable(Alexnet main.cu)

set_target_properties(Alexnet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)