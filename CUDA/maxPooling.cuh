#ifndef ALEXNET_MAXPOOLING_H
#define ALEXNET_MAXPOOLING_H
#include "../CUDA_or_CPU.cuh"
//__global__ void max_pooling_CUDA(float *image, float *res, int image_size, int pool_size, int stride,  int channel, int res_dim);
float* max_pooling(float *image, int image_size, int pool_size, int stride, int channel);
float* max_pooling_CPU(float *image, int img_size, int pool_size, int stride, int channel);

#endif