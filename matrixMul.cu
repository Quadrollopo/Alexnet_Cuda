// System includes
#include <stdio.h>

// CUDA runtime
#include "matrixMul.cuh"
#include <cuda_runtime.h>




__global__ void matrixMul(float *a, float *b, float *c, int b_row, int b_col) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    if(tx<b_row) {
        float x = a[bx*b_row+tx] * b[tx*b_col+by];

        __syncthreads();

        atomicAdd(&c[bx*b_col+by], x);
    }

}
/**
 * @param values first matrix (1 x weights_row)
 * @param weights second matrix (weights_row x weights_col as array)
 * @param weights_row rows of the second matrix
 * @param weights_col column of the second matrix
 * float *values, float *weights, int weights_row, int weights_col
 */
float* matrix_mul(float *values, float *weights, float *bias, int weights_row, int weights_col) {

    float *d_values, *d_weights, *d_res;

    auto res = new float[weights_col];

    for(int i=0;i<weights_col;i++)
        res[i] = 0.0f;


    cudaMalloc(&d_values, weights_row * sizeof(float));
    cudaMalloc(&d_weights, weights_row * weights_col * sizeof(float));
    cudaMalloc(&d_res, weights_col * sizeof(float));

    cudaMemcpy(d_values, values, weights_row * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weights_row * weights_col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, weights_col * sizeof(float), cudaMemcpyHostToDevice);

    matrixMul<<<dim3(1,weights_col), weights_row>>>(d_values,d_weights,d_res,weights_row,weights_col);

    cudaMemcpy(res, d_res, weights_col * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_weights);
    cudaFree(d_res);


    cudaDeviceReset();

    return res;
}

