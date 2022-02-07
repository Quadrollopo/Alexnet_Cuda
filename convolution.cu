#include "convolution.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void convolution_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if(tx * kernel_size + ty < kernel_size * kernel_size) {

        //controllare bx e by
        int kernel_left = by * stride - pad;
        int kernel_right = kernel_left + kernel_size - 1;
        int kernel_up = bx * stride - pad;
        int kernel_down = kernel_up + kernel_size - 1;

        float x;

        if((kernel_left < 0 && ty < pad) || //padding a sinistra
        (kernel_right >= image_size && ty >= kernel_size - pad) || //padding a destra
        (kernel_up < 0 && tx < pad) || //padding sopra
        (kernel_down >= image_size && tx >= kernel_size - pad)) //padding sotto
            x = 0.0f;

        else{
            int index = ( kernel_up + (kernel_size - 1)/2 ) * image_size + ( kernel_left + (kernel_size - 1)/2 ); // indice centrale
            int offset =  index + ( tx - (kernel_size - 1)/2) * image_size + ty - (kernel_size - 1)/2; // offset da aggiungere  o sottrarre
            x = image[index+offset] * kernel[tx * kernel_size + ty] / (float)kernel_size;
        }
        __syncthreads(); //??

        atomicAdd(&res[by*res_dim + bx], x);
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
    auto res_dim = (image_size-kernel_size+2*pad)/stride+1;
    auto res = new float[res_dim * res_dim];

    for(int i=0; i < res_dim * res_dim; i++)
        res[i] = 0.0f;


    cudaMalloc(&d_image, image_size * image_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_res, res_dim * res_dim * sizeof(float));

    cudaMemcpy(d_image, image, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, res_dim * res_dim * sizeof(float), cudaMemcpyHostToDevice);

    convolution_CUDA<<<dim3(res_dim, res_dim), kernel_size * kernel_size>>>(d_image, d_kernel, d_res, image_size, kernel_size, stride, pad, res_dim);

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
float* convolution_CPU(float *image, float *kernel, int kern_size, int img_size, int stride, int pad) {

	int kern_len = kern_size * kern_size;

	float* res = new float [(img_size - 1)*(img_size - 1)];

	for (int x=0; x < img_size - kern_size + 1; x+=img_size){
		for (int y=0; y < img_size - kern_size + 1; y++) {
			float sum = 0;
			for(int i=0; i<kern_size; i+=kern_size){
				for(int j=0; i<kern_size; j++){
					sum += kernel[i + j] * image[x + y];
				}
			}
		}
	}

    return res;
}
