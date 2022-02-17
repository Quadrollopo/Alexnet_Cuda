#include "utils.cuh"
#include <cuda_runtime.h>

__global__ void reLU(float *f, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
        float x = f[bx] > 0.0f ? f[bx] : 0.0f;
        f[bx] = x;
    }
}

void reLU_CUDA(float *f, int len){
    reLU<<<len, 1>>>(f, len);
}

__global__ void Heaviside(float *f, float *res, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
        res[bx] = f[bx] > 0.0f ? 1.0 : 0.0f;
    }
}

void  Heaviside_CUDA(float *f, float *res, int len){
    Heaviside<<<len, 1>>>(f, res, len);
}

__global__ void sigmoid(float *f, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
        f[bx] = 1.f/ (1.f + exp(-f[bx]));
    }
}

void sigmoid_CUDA(float *f, int len){
    sigmoid<<<len, 1>>>(f, len);
}

__global__ void der_sigmoid(float *f, float *res, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
        res[bx] = f[bx] * (1 - f[bx]);
    }
}

void  der_sigmoid_CUDA(float *f, float *res, int len){
    der_sigmoid<<<len, 1>>>(f, res, len);
}

__global__ void softmax(float *f, float sum, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
        float x = exp(f[bx]);
        atomicAdd(&sum,x);
        __syncthreads();
        f[bx] = x/sum;
    }
}

void Softmax_CUDA(float *f, int len){
    float sum = 0.f;
    softmax<<<len, 1>>>(f, sum, len);
}