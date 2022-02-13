#ifndef ALEXNET_CONVOLUTION_H
#define ALEXNET_CONVOLUTION_H
#include "../CUDA_or_CPU.cuh"
__global__ void convolution_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim, int image_ch, int kernel_ch);
float* convolution(float *image, float *kernel, int image_size, int kernel_size, int stride, int pad, int image_ch, int kernel_ch);
float* convolution_CPU(float *image, float *kernel, int kern_size, int img_size, int stride, bool pad);
#endif