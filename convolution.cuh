#ifndef ALEXNET_CONVOLUTION_H
#define ALEXNET_CONVOLUTION_H

__global__ void convolution_CUDA(float *a, float *b, float *c , int a_row, int rc, int col);
float* convolution(float *image, float *kernel, int a_row, int image_size, int kernel_size, int stride, int pad);
float* convolution_CPU(float *a, float *b, int a_row, int b_row, int b_col);

#endif