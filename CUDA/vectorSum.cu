
// CUDA runtime
#include "vectorSum.cuh"
#include <cuda_runtime.h>

__global__ void vector_sum_CUDA(float *a, float *b, int len){
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < len)
        a[id] += b[id];

}


void vector_sum(float *a, float *b,  int len){
    vector_sum_CUDA<<<len, 1>>>(a, b, len);

}

float* vector_sum_CPU(float *a, float *b, int len){
    auto res = new float[len];
    for (int i=0; i<len; i++){
        res[i]=a[i] + b[i];
    }
    return res;
}


__global__ void vector_mul_CUDA(float *a, float *b, float *c, int len){
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < len)
        c[id] = a[id] * b[id];

}


void vector_mul(float *a, float *b, float *c, int len){
    vector_mul_CUDA<<<len, 1>>>(a, b,c, len);

}

__global__ void vector_constant_mul_CUDA(float *a, float b, int len){
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < len)
        a[id] *= b;

}

void vector_constant_mul(float *a, float b, int len){
    vector_constant_mul_CUDA<<<len, 1>>>(a, b, len);
}

__global__ void vector_diff_CUDA(float *a, float *b, int len){
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < len)
        a[id] -= b[id];

}


void vector_diff(float *a, float *b, int len){
    vector_diff_CUDA<<<len, 1>>>(a, b, len);
}

__global__ void vector_diff_alloc_CUDA(const float *a, const float *b, float *c, int len){
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < len)
        c[id] = a[id] - b[id];

}


void vector_diff_alloc(const float *a, const float *b, float *c,  int len){
    vector_diff_alloc_CUDA<<<len, 1>>>(a, b,c, len);
}

__global__ void vector_conv_bias_CUDA(float *a, float *b, int num_sum){
	unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

#pragma unroll
	for(int i=0; i<num_sum; i++){
		a[id] += b[i + id*num_sum];
	}
}

void vector_conv_bias(float *a, float *b, int num_sum, int len_bias){
	vector_conv_bias_CUDA<<<len_bias, 1>>>(a, b, num_sum);
}

__global__ void loss_cross_entropy_der_CUDA(const float *out, const float* exp, float *res, int len){
	unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
	if (id < len){
		res[id] = (-1.f/(float)len) * ((exp[id]/(out[id] + 1e-10f)) - ((1 - exp[id])/(1 - out[id] + 1e-10f)));
	}
}


void loss_cross_entropy_der(const float *cost, const float* exp, float *res, int len){
	loss_cross_entropy_der_CUDA<<<1, len>>>(cost, exp, res, len);
}


__global__ void vector_bias_sum_CUDA(float *a, float *b){
	unsigned int id = blockIdx.x+blockDim.x*threadIdx.x;

	a[id] += b[threadIdx.x];
}

void vector_bias_sum(float *a, float *b,  int len, int num_bias){
	vector_bias_sum_CUDA<<<len, num_bias>>>(a,b);
}