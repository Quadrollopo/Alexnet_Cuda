#ifndef ALEXNET_VECTORSUM_H
#define ALEXNET_VECTORSUM_H
#include "../CUDA_or_CPU.cuh"
void vector_sum(float *a, float *b, int len);
float* vector_sum_CPU(float *a, float *b, int len);

#endif