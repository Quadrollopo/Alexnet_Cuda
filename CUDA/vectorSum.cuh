#ifndef ALEXNET_VECTORSUM_H
#define ALEXNET_VECTORSUM_H

__global__ void vector_sum_CUDA(float *a, float *b, int len);
void vector_sum(float *a, float *b, int len);
float* vector_sum_CPU(float *a, float *b, int len);

#endif