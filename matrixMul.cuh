#ifndef ALEXNET_MATRIXMUL_H
#define ALEXNET_MATRIXMUL_H

__global__ void matrixMul(float *a, float *b, float *c, int rc, int col);
float* matrix_mul(float *a, float *b, int a_row, int b_row, int b_col);
#endif