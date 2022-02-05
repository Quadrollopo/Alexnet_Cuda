#include "convolution.cuh"
#include <cuda_runtime.h>

__global__ void convolution_CUDA(float *a, float *b, float *c, int a_row, int b_row, int b_col) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    if(tx < kernel_size * kernel_size) { // Ha senso tenere kernel_size o è più comodo riceverlo già moltiplicato?
        float x = image[] * kernel[tx];
        __syncthreads(); //??

        atomicAdd(&c[bx*b_col+by], x);
    }

}
/**
 * @param image first matrix
 * @param kernel second matrix
 * @param a_row rows of the first matrix
 * @param b_row rows of the second matrix
 * @param b_col column of the second matrix
 * float *values, float *weights, int weights_row, int weights_col
 **/
float* convolution(float *image, float *kernel, int image_size, int kernel_size, int stride, int pad) {
    if(kernel_size % 2 == 0){
        std::cout << "Filter size is not odd" << std::endl;
        return nullptr;
    }
    if(pad > (kernel_size-1)/2){
        std::cout << "Pad is too high" << std::endl;
        return nullptr;
    }

    float *d_image, *d_kernel, *d_res;
    auto res_dim = (image_size-kernel_size+2*pad)/(stride+1);
    auto res = new float[res_dim * res_dim];

    for(int i=0; i < res_dim * res_dim; i++)
        res[i] = 0.0f;


    cudaMalloc(&d_image, image_size * image_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_res, res_dim * res_dim * sizeof(float));

    cudaMemcpy(d_image, image, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, res_dim * res_dim * sizeof(float), cudaMemcpyHostToDevice);

    convolution_CUDA<<<dim3(res_dim, res_dim), kernel_size * kernel_size>>>(d_image, d_kernel, d_res, image_size, kernel_size, stride, pad);

    cudaMemcpy(res, d_res, res_dim * res_dim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_res);

//    for(int i=0; i < a_row * b_col; i++){
//        printf("%f ", res[i]);
//    }
//    printf("\n\n\n\n");


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
float* convolution_CPU(float *a, float *b, int a_row, int b_row, int b_col, int size, int stride, int pad) {

    auto res = new float[a_row * b_col];

    for(int i=0; i < a_row * b_col; i++)
        res[i] = 0.0f;

    for(int i = 0; i < a_row; i++)
        for(int j=0; j < b_col; j++)
            for(int k = 0; k < b_row; k++)
                res[i*b_col+j] += a[i*b_row+k] * b[k*b_col+j] ;

    return res;
}
