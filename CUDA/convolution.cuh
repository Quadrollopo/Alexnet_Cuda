#ifndef ALEXNET_CONVOLUTION_H
#define ALEXNET_CONVOLUTION_H
#include <cuda_runtime.h>
#include <iostream>

__global__ void convolution_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim, int image_ch, int kernel_ch);
void convolution(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int image_ch, int kernel_ch);
void convolution_weights(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int image_ch, int kernel_ch);
__global__ void convolution_weights_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim, int image_ch, int kernel_ch);
void convolution_prevlayer_backpropagation(float *cost, float *kernel, float *res, int cost_size, int kernel_size, int prevlayer_size, int kernel_ch, int prevlayer_ch);
float* convolution_CPU(float *image, float *kernel, int img_size, int kern_size, int stride, int pad_size, int channels, int num_kernel);
float* convolution_cost_CPU(float *image, float *kernel, int img_size, int kern_size, int stride, int pad_size, int channels, int num_kernel);
float* convolution_weights_CPU(float *image, float *kernel, int img_size, int kern_size, int stride, int pad_size, int channels, int num_kernel);
#endif