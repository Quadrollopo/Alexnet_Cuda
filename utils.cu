#include "utils.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <string>

__global__ void reLU(float *f, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
        float x = f[bx] > 0.0f ? f[bx] : 0.0f;
        f[bx] = x;
    }
}

void reLU_CUDA(float *f, int len){
    reLU<<<len, 1>>>(f, len);
}

__global__ void Heaviside(float *f, float *res, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
        res[bx] = f[bx] > 0.0f ? 1.0 : 0.0f;
    }
}

void  Heaviside_CUDA(float *f, float *res, int len){
    Heaviside<<<len, 1>>>(f, res, len);
}

__global__ void sigmoid(float *f, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
        f[bx] = 1.f/ (1.f + exp(-f[bx]));
    }
}

void sigmoid_CUDA(float *f, int len){
	sigmoid<<<len, 1>>>(f, len);
}

__global__ void der_sigmoid(float *f, float *res, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
        res[bx] = f[bx] * (1 - f[bx]);
    }
}

void  der_sigmoid_CUDA(float *f, float *res, int len){
    der_sigmoid<<<len, 1>>>(f, res, len);
}

__global__ void softmax(float *f, float *sum_max, int len){
    unsigned int bx = blockIdx.x*blockDim.x+threadIdx.x;
    if(bx < len){
		if(bx == 0){
			sum_max[1] = f[0];
			for(int i=1; i<len; i++)
				if(f[i] > sum_max[1]) sum_max[1] = f[i];
		}
		__syncthreads();
		float x = f[bx] - sum_max[1];
		x = exp(x) + 1e-8f;
		atomicAdd(&sum_max[0], x);
		__syncthreads();

		f[bx] = x/(sum_max[0]);

    }
}

__global__ void pc(const float* a, int len){
	for(int i=0; i < len; i++){
		printf("%f ", a[i]);
		if(i % 10 == 9)
			printf("\n");
	}
	printf("\n");
	printf("\n");
}

void print_CUDA(const float* a, int len){
	pc<<<1,1>>>(a, len);
}

void Softmax_CUDA(float *f, int len){
	float *sum_max;
	cudaMalloc(&sum_max, 2*sizeof(float));
	cudaMemset(sum_max, 0,  2*sizeof(float));
	softmax<<<1, len>>>(f, sum_max, len);
}


void save_CUDA(const float *a, int len, int size, int z) {
	float *b = new float[len];
	cudaMemcpy(b, a, len*sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream f("save_" + std::to_string(z) + ".txt");
	for(int i=0; i < len; i++){

		f << b[i] << " ";
		if(i % size == size-1)
			f << std::endl;
	}
	f.close();
	delete[] b;
}

