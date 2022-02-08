#include "convolution.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void max_pooling_CUDA(float *image, float *res, int image_size, int kernel_size, int stride, int pad) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;


}

/**
 * @param image first matrix
 * @param kernel second matrix
 * @param image_size size of image
 * @param kernel_size size of kernel
 * @param stride
 * @param pad
 **/

float* max_pooling(float *image, int image_size, int kernel_size, int stride, int pad) {
    if(kernel_size % 2 == 0){
        std::cout << "Filter size is not odd" << std::endl;
        return nullptr;
    }
    if(pad > (kernel_size-1)/2){
        std::cout << "Pad is too high" << std::endl;
        return nullptr;
    }

    float *d_image, *d_res;
    int res_dim = (image_size-kernel_size+2*pad)/stride+1;
    float* res = new float[res_dim * res_dim];

    for(int i=0; i < res_dim * res_dim; i++)
        res[i] = 0.0f;
    int *kernel ;

    cudaMalloc(&d_image, image_size * image_size * sizeof(float));
    cudaMalloc(&d_res, res_dim * res_dim * sizeof(float));

    cudaMemcpy(d_image, image, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, res_dim * res_dim * sizeof(float), cudaMemcpyHostToDevice);

    max_pooling_CUDA<<<1,1>>>(image,res,image_size,kernel_size,stride,)

    cudaMemcpy(res, d_res, res_dim * res_dim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_res);

    printf("convolution GPU:\n");
    for(int i=0; i < res_dim * res_dim; i++){
        printf("%.2f ", res[i]);
    }
    printf("\n\n\n");


    cudaDeviceReset();

    return res;
}



/**
 * @param a first matrix (1 x weights_row)
 * @param b second matrix (weights_row x weights_col as array)
 * @param a_row rows of the first matrix
 * @param b_row rows of the second matrix
 * @param b_col column of the second matrix
 * float *values, float *weights, int weights_row, int weights_col
 */

float* max_pooling_CPU(float *image, float *kernel, int kern_size, int img_size, int stride, bool pad) {
    return nullptr;

}
