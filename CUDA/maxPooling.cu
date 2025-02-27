#include <cuda_runtime.h>
#include "maxPooling.cuh"

__global__ void max_pooling_CUDA(float *image, float *res, int *res2, int image_size, int pool_size, int stride,  int channel, int res_dim) {

	// Block index
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;

	// Thread index
	unsigned int tx = threadIdx.x;

	if(tx  < channel) {
		float x;
		float max = -3.40282347e+38; //lowest float
		int index_max = 0;

		int index = bx * stride * image_size + by * stride + tx * image_size * image_size; // indice iniziale

		for(int i = 0; i < pool_size; i++){
			for(int j = 0; j < pool_size; j++){
				x = image[index + i * image_size + j];
				if(x>max) {
					max=x;
					index_max = index + i * image_size + j;
				}
			}
		}
		res[bx * res_dim + by + tx * res_dim * res_dim] = max;
		res2[bx * res_dim + by + tx * res_dim * res_dim] = index_max;
	}
}

/**
 * @param image first matrix
 * @param kernel second matrix
 * @param image_size size of image
 * @param kernel_size size of kernel
 * @param stride
 * @param pad
 **/

void max_pooling(float *image, float *res, int *res2, int image_size, int pool_size, int stride, int channel) {
	int res_dim = (image_size-pool_size)/stride+1;
	max_pooling_CUDA<<<dim3(res_dim, res_dim),channel>>>(image,res,res2,image_size,pool_size,stride,channel,res_dim);
}

__global__ void max_unpooling_CUDA(float *cost, int *max_indexes, float *res, int input_size, int channel) {

	unsigned int id = blockIdx.x * input_size + blockIdx.y + threadIdx.x * input_size * input_size;

	if(id  < input_size * input_size * channel) {
		res[max_indexes[id]] = cost[id];
	}
}

void max_unpooling(float *cost, int *max_indexes, float *res, int input_size, int output_size, int channel) {
	//we are going backward, so input comes from right and output is on left, with output_size > input_size
	cudaMemset(res, 0 , output_size * output_size * channel);
	max_unpooling_CUDA<<<dim3(input_size, input_size),channel>>>(cost, max_indexes, res, input_size, channel);
}


__global__ void max_pooling_CUDA(float *image, float *res, int image_size, int pool_size, int stride,  int channel, int res_dim) {

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;

	if(tx  < channel) {
		float x;
		float max = -3.40282347e+38;

		int index = bx * stride * image_size + by * stride + tx * image_size * image_size; // indice iniziale

		for(int i = 0; i < pool_size; i++){
			for(int j = 0; j < pool_size; j++){
				x = image[index + i * image_size + j];
				if(x>max)
					max=x;
			}
		}
		res[bx * res_dim + by + tx * res_dim * res_dim] = max;
	}
}

/**
 * @param image first matrix
 * @param kernel second matrix
 * @param image_size size of image
 * @param kernel_size size of kernel
 * @param stride
 * @param pad
 **/


float* max_pooling_CPU(float *image, int img_size, int pool_size, int stride, int channel) {
	int res_size = (img_size - pool_size) / stride + 1;
	float* res = new float [res_size * res_size];
	for (int x=0, x_image=0; x < res_size; x++, x_image+=stride){
		for (int y=0, y_image=0; y < res_size; y++, y_image+=stride) {
			int res_index = x * res_size + y;
			for (int i = 0; i < pool_size; i++) {
				for (int j = 0; j < pool_size; j++) {
					int img_index = (x_image + i) * img_size + y_image + j;
					if(res[res_index] < image[img_index])
						res[res_index] = image[img_index];
				}
			}
		}
	}
	return res;

}



