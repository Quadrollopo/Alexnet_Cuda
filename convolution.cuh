#ifndef ALEXNET_CONVOLUTION_H
#define ALEXNET_CONVOLUTION_H

__global__ void convolution_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim);
float* convolution(float *image, float *kernel, int image_size, int kernel_size, int stride, int pad);
float* convolution_CPU(float *image, float *kernel, int kern_size, int img_size, int stride, bool pad);

#endif