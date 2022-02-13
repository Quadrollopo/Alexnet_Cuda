#include "utils.cuh"
#include "stdio.h"
#include <cuda_runtime.h>

__global__ void reLU(float *f, int len){
    unsigned int bx = threadIdx.x;
    if(bx < len){
        float x = f[bx] > 0.0f ? f[bx] : 0.0f;
        f[bx] = x;
    }
}

void reLU_CUDA(float *f, int len){
    reLU<<<1, len>>>(f, len);
}

__global__ void Heaviside(float *f, float *res, int len){
    unsigned int bx = threadIdx.x;
    if(bx < len){
        res[bx] = f[bx] > 0.0f ? 1.0 : 0.0f;
    }
}

float*  Heaviside_CUDA(float *f, int len){
    float *res;
    cudaMalloc(&res, len * sizeof(float) );
    Heaviside<<<1, len>>>(f, res, len);
    return res;
}

__global__ void sigmoid(float *f, int len){
    unsigned int bx = threadIdx.x;
    if(bx < len){
        f[bx] = 1.f/ (1.f + exp(-f[bx]));
    }
}

void sigmoid_CUDA(float *f, int len){
    sigmoid<<<1, len>>>(f, len);
}

__global__ void der_sigmoid(float *f, float *res, int len){
    unsigned int bx = threadIdx.x;
    if(bx < len){
        res[bx] = f[bx] * (1 - f[bx]);
    }
}

float*  der_sigmoid_CUDA(float *f, int len){
    float *res;
    cudaMalloc(&res, len * sizeof(float) );
    der_sigmoid<<<1, len>>>(f, res, len);
    return res;
}

__global__ void softmax(float *f, float sum, int len){
    unsigned int bx = threadIdx.x;
    if(bx < len){
        float x = exp(f[bx]);
        atomicAdd(&sum,x);
        __syncthreads();
        f[bx] = x/sum;
    }
}

void Softmax_CUDA(float *f, int len){
    float sum = 0.f;
    softmax<<<1, len>>>(f, sum, len);
}

void nope(float *f, int len){
    int x = len;
}

float* nope_der(float *f, int len){
    return f;
}