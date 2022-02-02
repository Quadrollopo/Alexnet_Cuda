cmake_minimum_required(VERSION 3.10)
project(Alexnet CUDA)

set(CMAKE_CUDA_STANDARD 14)
add_executable(Alexnet main.cu network.cu network.h fullLayer.cu fullLayer.h)

set_target_properties(Alexnet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)