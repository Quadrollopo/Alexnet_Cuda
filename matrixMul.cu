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
    if(tx < b_row) {
        float x = a[bx*b_row+tx] * b[tx*b_col+by];

        __syncthreads();

        atomicAdd(&c[bx*b_col+by], x);
    }

}
/**
 * @param a first matrix (1 x weights_row)
 * @param b second matrix (weights_row x weights_col as array)
 * @param b_row rows of the second matrix
 * @param b_col column of the second matrix
 * float *values, float *weights, int weights_row, int weights_col
 */
float* matrix_mul(float *a, float *b, int a_row, int b_row, int b_col) {

    float *d_a, *d_b, *d_c;

    auto res = new float[a_row * b_col];

    for(int i=0; i < a_row * b_col; i++)
        res[i] = 0.0f;


    cudaMalloc(&d_a, a_row * b_row * sizeof(float));
    cudaMalloc(&d_b, b_row * b_col * sizeof(float));
    cudaMalloc(&d_c, a_row * b_col * sizeof(float));

    cudaMemcpy(d_a, a, a_row * b_row * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_row * b_col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, res, a_row * b_col * sizeof(float), cudaMemcpyHostToDevice);

    matrixMul<<<dim3(a_row, b_col), b_row>>>(d_a, d_b, d_c, b_row, b_col);

    cudaMemcpy(res, d_c, a_row * b_col * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for(int i=0; i < a_row * b_col; i++){
        printf("%f ", res[i]);
    }


    cudaDeviceReset();

    return res;
}

