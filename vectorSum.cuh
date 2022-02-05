#ifndef ALEXNET_VECTORSUM_H
#define ALEXNET_VECTORSUM_H

__global__ void vectorSum(float *a, float *b, float *c, int len);
float* vector_sum(float *a, float *b, int len);
float* vector_sumCPU(float *a, float *b, int len);

#endif