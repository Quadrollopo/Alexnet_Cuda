#include "convolution.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

#if CUDA

__global__ void max_pooling_CUDA(float *image, float *res, int image_size, int pool_size, int stride,  int channel, int res_dim) {

    // Block index
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    // Thread index
    unsigned int tx = threadIdx.x;

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

float* max_pooling(float *image, int image_size, int pool_size, int stride, int channel) {
//    if(pool_size % 2 == 0){
//        std::cout << "Filter size is not odd" << std::endl;
//        return nullptr;
//    }
    float *d_image, *d_res;
    int res_dim = (image_size-pool_size)/stride+1;
    float* res = new float[res_dim * res_dim * channel]();


    cudaMalloc(&d_image, image_size * image_size * channel * sizeof(float));
    cudaMalloc(&d_res, res_dim * res_dim * channel * sizeof(float));

    cudaMemcpy(d_image, image, image_size * image_size * channel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, res_dim * res_dim * channel * sizeof(float), cudaMemcpyHostToDevice);

    max_pooling_CUDA<<<dim3(res_dim, res_dim),channel>>>(d_image,d_res,image_size,pool_size,stride,channel,res_dim);

    cudaMemcpy(res, d_res, res_dim * res_dim * channel * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_res);

    cudaDeviceReset();

    return res;
}


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


#else

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

float* max_pooling(float *image, int image_size, int pool_size, int stride, int channel) {
//    if(pool_size % 2 == 0){
//        std::cout << "Filter size is not odd" << std::endl;
//        return nullptr;
//    }
    float *d_image, *d_res;
    int res_dim = (image_size-pool_size)/stride+1;
    float* res = new float[res_dim * res_dim * channel]();


    cudaMalloc(&d_image, image_size * image_size * channel * sizeof(float));
    cudaMalloc(&d_res, res_dim * res_dim * channel * sizeof(float));

    cudaMemcpy(d_image, image, image_size * image_size * channel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, res_dim * res_dim * channel * sizeof(float), cudaMemcpyHostToDevice);

    max_pooling_CUDA<<<dim3(res_dim, res_dim),channel>>>(d_image,d_res,image_size,pool_size,stride,channel,res_dim);

    cudaMemcpy(res, d_res, res_dim * res_dim * channel * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_res);

    cudaDeviceReset();

    return res;
}


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


#endif


