// System includes
#include <stdio.h>

// CUDA runtime
#include "vectorSum.cuh."
#include <cuda_runtime.h>


__global__ void vectorSum(float *a, float *b, float *c, int len){
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < len)
        c[id] = a[id] + b[id];

}


float* vector_sum(float *a, float *b, int len){
    float *d_a, *d_b, *d_c;

    auto res = new float[len];

    for(int i=0; i < len; i++)
        res[i] = 0.0f;


    cudaMalloc(&d_a, len * sizeof(float));
    cudaMalloc(&d_b, len * sizeof(float));
    cudaMalloc(&d_c, len * sizeof(float));

    cudaMemcpy(d_a, a, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, res, len * sizeof(float), cudaMemcpyHostToDevice);

    vectorSum<<<1, len>>>(d_a, d_b, d_c, len);

    cudaMemcpy(res, d_c, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

//    for(int i=0; i < a_row * b_col; i++){
//        printf("%f ", res[i]);
//    }
//    printf("\n\n\n\n");


    cudaDeviceReset();

    return res;
}

float* vector_sumCPU(float *a, float *b, int len){
    auto res = new float[len];
    for (int i=0; i<len; i++){
        res[i]=a[i] + b[i];
    }
    return res;
}