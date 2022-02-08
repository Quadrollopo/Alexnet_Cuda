#ifndef ALEXNET_MAXPOOLING_H
#define ALEXNET_MAXPOOLING_H

__global__ void max_pooling_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim);
float* max_pooling(float *image, float *kernel, int image_size, int kernel_size, int stride, int pad);
float* max_pooling_CPU(float *image, float *kernel, int kern_size, int img_size, int stride, bool pad);

#endif