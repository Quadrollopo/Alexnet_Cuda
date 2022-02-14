// System includes
#include <stdio.h>

// CUDA runtime
#include "vectorSum.cuh"
#include <cuda_runtime.h>

#if CUDA
__global__ void vector_sum_CUDA(float *a, float *b, int len){
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < len)
        a[id] += b[id];

}


void vector_sum(float *a, float *b, int len){

    vector_sum_CUDA<<<1, len>>>(a, b, len);

//    for(int i=0; i < a_row * b_col; i++){
//        printf("%f ", res[i]);
//    }
//    printf("\n\n\n\n");

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


float* vector_mul(float *a, float *b, int len){
    float *d_res;
    cudaMalloc(&d_res, len * sizeof(float));
    cudaMemset(d_res,0,len * sizeof(float));

    vector_mul_CUDA<<<1, len>>>(a, b,d_res, len);

    return d_res;

}

__global__ void vector_constant_mul_CUDA(float *a, float b, int len){
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < len)
        a[id] *= b;

}

void vector_constant_mul(float *a, float b, int len){

    vector_constant_mul_CUDA<<<1, len>>>(a, b, len);
}

__global__ void vector_diff_CUDA(float *a, float *b, int len){
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < len)
        a[id] -= b[id];

}


void vector_diff(float *a, float *b, int len){
    vector_diff_CUDA<<<1, len>>>(a, b, len);
}

__global__ void vector_diff_alloc_CUDA(const float *a, const float *b, float *c, int len){
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < len)
        c[id] = a[id] - b[id];

}


float* vector_diff_alloc(const float *a, const float *b, int len){
    float *d_res;
    cudaMalloc(&d_res, len * sizeof(float));
    cudaMemset(d_res,0,len * sizeof(float));
    vector_diff_alloc_CUDA<<<1, len>>>(a, b,d_res, len);
    return d_res;
}

#else
__global__ void vector_sum_CUDA(float *a, float *b, int len){
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < len)
        a[id] += b[id];

}


void vector_sum(float *a, float *b, int len){
    float *d_a, *d_b;


    cudaMalloc(&d_a, len * sizeof(float));
    cudaMalloc(&d_b, len * sizeof(float));

    cudaMemcpy(d_a, a, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, len * sizeof(float), cudaMemcpyHostToDevice);

    vector_sum_CUDA<<<1, len>>>(d_a, d_b, len);

    cudaMemcpy(a, d_a, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);

//    for(int i=0; i < a_row * b_col; i++){
//        printf("%f ", res[i]);
//    }
//    printf("\n\n\n\n");


    cudaDeviceReset();
}

float* vector_sum_CPU(float *a, float *b, int len){
    auto res = new float[len];
    for (int i=0; i<len; i++){
        res[i]=a[i] + b[i];
    }
    return res;
}



#endif
