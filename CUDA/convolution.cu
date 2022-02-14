#include "convolution.cuh"
#include <cuda_runtime.h>
#include <iostream>
#if CUDA

__global__ void convolution_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim,int image_ch, int kernel_ch) {

    // Block index
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    // Thread index
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    if(tx * kernel_size + ty < kernel_size * kernel_size) {
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
            int offset = ( tx - (kernel_size - 1)/2) * image_size + ty - (kernel_size - 1)/2; // offset da aggiungere  o sottrarre

            for(int i = 0; i < image_ch; i++){
                for (int j = 0; j < kernel_ch; j++){
                    x = image[index + offset + i * image_size * image_size]  *
                        kernel[tx * kernel_size + ty + j * kernel_size * kernel_size * image_ch + i * kernel_size * kernel_size];
                    atomicAdd(&res[bx * res_dim + by + j * res_dim * res_dim], x);
                }
            }
        }

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
float* convolution(float *image, float *kernel, int image_size, int kernel_size, int stride, int pad, int image_ch, int kernel_ch) {
    if(kernel_size % 2 == 0){
        std::cout << "Filter size is not odd" << std::endl;
        return nullptr;
    }
    if(pad > (kernel_size-1)/2){
        std::cout << "Pad is too high" << std::endl;
        return nullptr;
    }

    float *d_res;
    int res_dim = (image_size-kernel_size+2*pad)/stride+1;
    float* res = new float[res_dim * res_dim * kernel_ch]();


    cudaMalloc(&d_res, res_dim * res_dim * kernel_ch * sizeof(float));

    cudaMemcpy(d_res, res, res_dim * res_dim * kernel_ch * sizeof(float), cudaMemcpyHostToDevice);

    convolution_CUDA<<<dim3(res_dim, res_dim), dim3(kernel_size, kernel_size)>>>(image, kernel, d_res, image_size, kernel_size, stride, pad, res_dim, image_ch, kernel_ch);
//
    cudaMemcpy(res, d_res, res_dim * res_dim *  kernel_ch * sizeof(float), cudaMemcpyDeviceToHost);
//
//    cudaFree(d_res);

    printf("convolution GPU:\n");
    for(int i=0; i < kernel_ch; i++){
        for(int j=0; j < res_dim * res_dim; j++)
            printf("%d ", (int)res[i*res_dim*res_dim + j]);
        printf("\n");
    }
    printf("\n\n\n");
    delete[] res;


    return d_res;
}
/*
__global__ void convolution_backpropagation_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim,int image_ch, int kernel_ch) {

    // Block index
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    // Thread index
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    if(tx * kernel_size + ty < kernel_size * kernel_size) {
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
            int offset = ( tx - (kernel_size - 1)/2) * image_size + ty - (kernel_size - 1)/2; // offset da aggiungere  o sottrarre

            for(int i = 0; i < image_ch; i++){
                for (int j = 0; j < kernel_ch; j++){
                    x = image[index + offset + i * image_size * image_size]  *
                        kernel[tx * kernel_size + ty + j * kernel_size * kernel_size * image_ch + i * kernel_size * kernel_size];
                    atomicAdd(&res[bx * res_dim + by + j * res_dim * res_dim], x);
                }
            }
        }

    }

}

float* convolution_backpropagation(float *cost, float *kernel, int cost_size, int kernel_size, int stride, int pad, int cost_ch, int kernel_ch) {
    if(kernel_size % 2 == 0){
        std::cout << "Filter size is not odd" << std::endl;
        return nullptr;
    }
    if(pad > (kernel_size-1)/2){
        std::cout << "Pad is too high" << std::endl;
        return nullptr;
    }

    float *d_res;
    int res_dim = (cost_size ) * stride;
    float* res = new float[res_dim * res_dim * kernel_ch]();


    cudaMalloc(&d_res, res_dim * res_dim * kernel_ch * sizeof(float));

    cudaMemcpy(d_res, res, res_dim * res_dim * kernel_ch * sizeof(float), cudaMemcpyHostToDevice);

    convolution_CUDA<<<dim3(res_dim, res_dim), dim3(kernel_size, kernel_size)>>>(image, kernel, d_res, image_size, kernel_size, stride, pad, res_dim, image_ch, kernel_ch);
//
    cudaMemcpy(res, d_res, res_dim * res_dim *  kernel_ch * sizeof(float), cudaMemcpyDeviceToHost);
//
//    cudaFree(d_res);

    printf("convolution GPU:\n");
    for(int i=0; i < kernel_ch; i++){
        for(int j=0; j < res_dim * res_dim; j++)
            printf("%d ", (int)res[i*res_dim*res_dim + j]);
        printf("\n");
    }
    printf("\n\n\n");
    delete[] res;


    //return res;
    return d_res;
}*/



float* convolution_CPU(float *image, float *kernel, int kern_size, int img_size, int stride, bool pad) {
    int depth = 2;
    int pad_size;
    if(pad) {
        pad_size = (kern_size - 1);
    }
    else{
        pad_size = 0;
    }
    int res_size = (img_size + pad_size - kern_size) / stride + 1;
    pad_size /= 2;
    float* res = new float [(res_size)*(res_size)];

    for (int x=0; x < res_size; x++){
        for (int y=0; y < res_size; y++) {
            int res_index = x * res_size + y;
            res[res_index] = 0;
            int x_image = x * stride - pad_size;
            int y_image = y * stride - pad_size;
            for (int i = 0; i < kern_size; i++) {
                for (int j = 0; j < kern_size; j++) {
                    for (int z=0; z < depth; z++) {
                        if (x_image + i < 0
                            || y_image + j < 0
                            || x_image + i > img_size - 1
                            || y_image + j > img_size - 1)
                            continue;
                        res[res_index] += kernel[i * kern_size + j + z*kern_size*kern_size] *
                                          image[(i + x_image) * img_size + j + y_image + z*img_size*img_size];
                    }
                }
            }
        }
    }

    return res;
}

#else

__global__ void convolution_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim,int image_ch, int kernel_ch) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if(tx * kernel_size + ty < kernel_size * kernel_size) {
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
            int offset = ( tx - (kernel_size - 1)/2) * image_size + ty - (kernel_size - 1)/2; // offset da aggiungere  o sottrarre

            for(int i = 0; i < image_ch; i++){
                for (int j = 0; j < kernel_ch; j++){
                        x = image[index + offset + i * image_size * image_size]  *
                                kernel[tx * kernel_size + ty + j * kernel_size * kernel_size * image_ch + i * kernel_size * kernel_size];
                        atomicAdd(&res[bx * res_dim + by + j * res_dim * res_dim], x);
                }
            }
        }

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
float* convolution(float *image, float *kernel, int image_size, int kernel_size, int stride, int pad, int image_ch, int kernel_ch) {
    if(kernel_size % 2 == 0){
        std::cout << "Filter size is not odd" << std::endl;
        return nullptr;
    }
    if(pad > (kernel_size-1)/2){
        std::cout << "Pad is too high" << std::endl;
        return nullptr;
    }

    float *d_image, *d_kernel, *d_res;
    int res_dim = (image_size-kernel_size+2*pad)/stride+1;
    float* res = new float[res_dim * res_dim * kernel_ch]();


    cudaMalloc(&d_image, image_size * image_size * image_ch * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * image_ch * kernel_ch * sizeof(float));
    cudaMalloc(&d_res, res_dim * res_dim * kernel_ch * sizeof(float));

    cudaMemcpy(d_image, image, image_size * image_size * image_ch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * image_ch * kernel_ch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, res_dim * res_dim * kernel_ch * sizeof(float), cudaMemcpyHostToDevice);

    convolution_CUDA<<<dim3(res_dim, res_dim), dim3(kernel_size, kernel_size)>>>(d_image, d_kernel, d_res, image_size, kernel_size, stride, pad, res_dim, image_ch,kernel_ch);

    cudaMemcpy(res, d_res, res_dim * res_dim *  kernel_ch * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_res);

    printf("convolution GPU:\n");
    for(int i=0; i < kernel_ch; i++){
        for(int j=0; j < res_dim * res_dim; j++)
            printf("%d ", (int)res[i*res_dim*res_dim + j]);
        printf("\n");
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



float* convolution_CPU(float *image, float *kernel, int img_size, int kern_size, int stride, int pad_size, int channel, int num_kernel) {
	int res_size = (img_size + pad_size * 2 - kern_size) / stride + 1;
	int kern_len = kern_size*kern_size*channel;
	float* res = new float [res_size * res_size * num_kernel];

	for (int n=0, w=0, off_img=0; w <num_kernel; w++, n+=kern_len, off_img+=res_size*res_size) {
		for (int x = 0; x < res_size; x++) {
			for (int y = 0; y < res_size; y++) {
				int res_index = off_img + (x * res_size + y);
				res[res_index] = 0;
				int x_image = x * stride - pad_size;
				int y_image = y * stride - pad_size;
				for (int i = 0; i < kern_size; i++) {
					for (int j = 0; j < kern_size; j++) {
						for (int z = 0; z < channel; z++) {
							if (x_image + i < 0
								|| y_image + j < 0
								|| x_image + i > img_size - 1
								|| y_image + j > img_size - 1)
								continue;
							res[res_index] += kernel[n + i * kern_size + j + z * kern_size * kern_size] *
											  image[(i + x_image) * img_size + j + y_image + z * img_size * img_size];
						}
					}
				}
			}
		}
	}

    return res;
}

float* convolution_weights_CPU(float *image, float *kernel, int img_size, int kern_size, int stride, int pad_size, int channel, int num_kernel) {
	int res_size = (img_size + pad_size * 2 - kern_size) / stride + 1;
	int res_len = res_size * res_size;
	float* res = new float [res_len * num_kernel * channel];

	for (int n=0, w=0; w <num_kernel; w++, n+=res_len*channel) {
		for (int z = 0; z < channel; z++) {
			for (int x = 0; x < res_size; x++) {
				for (int y = 0; y < res_size; y++) {
					int res_index = n + z * res_len + x * res_size + y;
					res[res_index] = 0;
					int x_image = x * stride - pad_size;
					int y_image = y * stride - pad_size;
					for (int i = 0; i < kern_size; i++) {
						for (int j = 0; j < kern_size; j++) {
							if (x_image + i < 0
								|| y_image + j < 0
								|| x_image + i > img_size - 1
								|| y_image + j > img_size - 1)
								continue;
							res[res_index] += kernel[w * kern_size*kern_size + i * kern_size + j] *
											  image[(i + x_image) * img_size + j + y_image + z * img_size * img_size];
						}
					}
				}
			}
		}
	}

	return res;
}

float* convolution_cost_CPU(float *image, float *kernel, int img_size, int kern_size, int stride, int pad_size, int channel, int num_kernel) {
	int res_size = (img_size + pad_size * 2 - kern_size) / stride + 1;
	int kern_len = kern_size*kern_size*channel;
	float* res = new float [res_size * res_size * channel];

	for (int z = 0; z < channel; z++) {
		for (int x = 0; x < res_size; x++) {
			for (int y = 0; y < res_size; y++) {
				int res_index = z + (x * res_size + y);
				res[res_index] = 0;
				int x_image = x * stride - pad_size;
				int y_image = y * stride - pad_size;
				for (int i = 0, rev_i = kern_size-1; i < kern_size; i++, rev_i--) {
					for (int j = 0, rev_j = kern_size-1; j < kern_size; j++, rev_j--) {
						for (int n=0, w=0; w <num_kernel; w++, n+=kern_len) {
							if (x_image + i < 0
								|| y_image + j < 0
								|| x_image + i > img_size - 1
								|| y_image + j > img_size - 1)
								continue;
							res[res_index] += kernel[n + rev_i * kern_size + rev_j + z * kern_size * kern_size] *
											  image[(i + x_image) * img_size + j + y_image + z * img_size * img_size];
						}
					}
				}
			}
		}
	}

	return res;
}

#endif

