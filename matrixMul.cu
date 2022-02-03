// System includes
#include <stdio.h>
#include <assert.h>

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
        int x = a[bx*b_row+tx] * b[tx*b_col+by];

        __syncthreads();

        atomicAdd(&c[bx*b_col+by], x);
    }

}
/**
 * @param values first matrix (1 x weights_row)
 * @param weights second matrix (weights_row x weights_col as array)
 * @param res results matrix (1 x weights_col)
 * @param weights_row rows of the second matrix
 * @param weights_col column of the second matrix
 * float *values, float *weights, float *res, int weights_row, int weights_col
 */
void matrix_mul(float *values, float *weights, float *res, int weights_row, int weights_col) {

    float *d_values, *d_weights, *d_res;


    cudaMalloc(&d_values, weights_row * sizeof(float));
    cudaMalloc(&d_weights, weights_row * weights_col * sizeof(float));
    cudaMalloc(&d_res, weights_col * sizeof(float));

    cudaMemcpy(d_values, values, weights_row * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weights_row * weights_col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, weights_col * sizeof(float), cudaMemcpyHostToDevice);

    matrixMul<<<dim3(1,weights_col), weights_row>>>(d_values,d_weights,d_res,weights_row,weights_col);

    cudaMemcpy(&res, d_res, weights_col * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<weights_col;i++)
        printf("%f ",res[i]);

    cudaDeviceReset();
}