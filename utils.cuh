#ifndef ALEXNET_UTLIS_H
#define ALEXNET_UTLIS_H
#include <stdio.h>
#include <iostream>
//CUDA
void reLU_CUDA(float *f, int len);
void Heaviside_CUDA(float *f, float *res, int len);
void sigmoid_CUDA(float *f, int len);
void der_sigmoid_CUDA(float *f, float *res, int len);
void Softmax_CUDA(float *input, int len);

void print_CUDA(const float* a, int len);
void save_CUDA(const float* a, int len, int size, int z);
#endif