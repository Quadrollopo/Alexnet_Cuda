#ifndef ALEXNET_MAXPOOLING_H
#define ALEXNET_MAXPOOLING_H
#include "../CUDA_or_CPU.cuh"
void max_pooling(float *image, float *res, int *res2, int image_size, int pool_size, int stride, int channel);
void max_unpooling(float *max, int *max_indexes, float *res, int input_size, int output_size, int channel);
float* max_pooling_CPU(float *image, int img_size, int pool_size, int stride, int channel);

#endif