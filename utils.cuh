#ifndef ALEXNET_UTLIS_H
#define ALEXNET_UTLIS_H
#include <stdio.h>
#include <iostream>
//CUDA
void reLU_CUDA(float *f, int len);
void Heaviside_CUDA(float *f, float *res, int len);
void sigmoid_CUDA(float *f, int len);
void der_sigmoid_CUDA(float *f, float *res, int len);
void Softmax_CUDA(float *input, int len);

static void print_CUD(const float* a, int len){
	float *res = new float[len];
    cudaMemcpy(res, a, len * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i < len; i++){
		std::cout << res[i] << " ";
		if(i % 10 == 9)
			std::cout << std::endl;
    }
	std::cout << std::endl;
    delete[] res;
}
#endif