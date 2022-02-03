// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include "matrixMul.cuh"
#include <cuda_runtime.h>




__global__ void matrixMul(int *a, int *b, int *c, int rc, int col) {

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

    for (int i = 0; i < 3; i++) {
        cudaMalloc(&d_a[i], 3 * sizeof(int));
        cudaMemcpy(d_a[i], a1, 3 * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&d_b, 3 * sizeof(int *));
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_b[i], 2 * sizeof(int));
        cudaMemcpy(d_b[i], b1, 6 * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&d_c, 4 * sizeof(int *));
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_c[i], 2 * sizeof(int));
        cudaMemcpy(d_c[i]
                , c1, 8 * sizeof(int), cudaMemcpyHostToDevice);
    }
*/
    cudaMalloc(&d_a, 12 * sizeof(int));
    cudaMalloc(&d_b, 6 * sizeof(int));
    cudaMalloc(&d_c, 8 * sizeof(int));
    cudaMemcpy(d_a, a1, 12 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b1, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c1, 8 * sizeof(int), cudaMemcpyHostToDevice);

    matrixMul<<<dim3(4,2), 3>>>(d_a,d_b,d_c,3,2);

    cudaMemcpy(&c1, d_c, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<8;i++)
        printf("%d ",c1[i]);
    cudaDeviceReset();
}