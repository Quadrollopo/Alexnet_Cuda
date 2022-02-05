// System includes
#include <stdio.h>

// CUDA runtime
#include "vectorSum.cuh."
#include <cuda_runtime.h>


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