cmake_minimum_required(VERSION 3.17)
project(Alexnet CUDA)

set(CMAKE_CUDA_STANDARD 14)
add_executable(Alexnet main.cu network.cu network.h FullLayer.cu FullLayer.h matrixMul.cu matrixMul.cuh)

set_target_properties(Alexnet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)