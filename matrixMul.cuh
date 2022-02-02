#ifndef ALEXNET_MATRIXMUL_H
#define ALEXNET_MATRIXMUL_H

__global__ void matrixMul(int **a, int **b, int **c);
void matrix_mul();
#endif