#ifndef ALEXNET_UTLIS_H
#define ALEXNET_UTLIS_H
#include "CUDA_or_CPU.cuh"

//CUDA
void reLU_CUDA(float *f, int len);
float* Heaviside_CUDA(float *f, int len);
void sigmoid_CUDA(float *f, int len);
float* der_sigmoid_CUDA(float *f, int len);
void Softmax_CUDA(float *input, int len);

static float reLU(float f){
	return f > 0.0f ? f : 0.0f;
}
static float Heaviside(float f){
    return f > 0.0f ? 1.0f : 0.0f;
}
static float sigmoid(float f){
    return 1.f/ (1.f + exp(-f));
}
static float der_sigmoid(float f) {
    return f * (1 - f);
}
static float* Softmax(float input[], int length){
    float sum = 0;
    for(int i = 0; i < length; i++) {
        input[i] = exp(input[i]);
        sum += input[i];
    }
    for(int i = 0; i < length; i++)
        input[i] = input[i]/sum;
    return input;
}
void nope(float *f, int len);
float* nope_der(float *f, int len);
#endif