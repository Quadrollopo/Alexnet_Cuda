#ifndef ALEXNET_MATRIXMUL_H
#define ALEXNET_MATRIXMUL_H

__global__ void matrixMul(float *a, float *b, float *c, int rc, int col);
void matrix_mul(float *values, float *weights, float *res, int weights_row, int weights_col);
#endif