#ifndef ALEXNET_VECTORSUM_H
#define ALEXNET_VECTORSUM_H

void vector_sum(float *a, float *b, int len);
float* vector_sum_CPU(float *a, float *b, int len);
void vector_mul(float *a, float *b, float *c, int len);
void vector_constant_mul(float *a, float b, int len);
void vector_diff(float *a, float *b, int len);
void vector_diff_alloc(const float *a, const float *b, float *c,  int len);
void loss_cross_entropy_der(const float *cost, const float* exp, float *res, int len);
void vector_conv_bias(float *a, float *b, int num_sum, int len);


#endif