#ifndef ALEXNET_MATRIXMUL_H
#define ALEXNET_MATRIXMUL_H
#include <cuda_runtime.h>


void matrix_mul(float *a, float *b, float *c, int a_row, int b_row, int b_col);
void matrix_mul2(float *a, float *b, float *c, int a_row, int b_row, int b_col);
void matrix_mul3(float *a, float *b, float *c, int a_row, int b_row, int b_col);
void matrix_mul_CPU(float *a, float *b, float *c, int a_row, int b_row, int b_col);
void printsum();



#endif
