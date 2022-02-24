#include "convolution.cuh"
#include "../utils.cuh"


__global__ void convolution_CUDA(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim,int image_ch, int kernel_ch) {

    // Block index
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    // Thread index
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    int kernel_len = kernel_size * kernel_size;
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
			return;

        else{
            int index = ( kernel_up + (kernel_size - 1)/2 ) * image_size + ( kernel_left + (kernel_size - 1)/2 ); // indice centrale
            int offset = ( tx - (kernel_size - 1)/2) * image_size + ty - (kernel_size - 1)/2; // offset da aggiungere  o sottrarre

            int image_len = image_size * image_size;
            int res_len = res_dim * res_dim;

            for(int i = 0; i < image_ch; i++){
                for (int j = 0; j < kernel_ch; j++){
                    x = image[index + offset + i * image_len]  *
                        kernel[tx * kernel_size + ty + j * kernel_len * image_ch + i * kernel_len];
                    atomicAdd(&res[bx * res_dim + by + j * res_len], x);
                }
            }
        }

    }

}

__global__ void convolution_weights_CUDA(
		float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int res_dim, int image_ch, int num_kernel) {

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	if(tx * kernel_size + ty < kernel_size * kernel_size) {
		int x_image = tx * stride - pad + bx;
		int y_image = ty * stride - pad + by;


		float z;
		if(y_image < 0
			|| x_image < 0
			|| y_image > image_size - 1
			|| x_image > image_size - 1){
			return;
		}
		else{

			for (int j = 0; j < num_kernel; j++){
				for(int i = 0; i < image_ch; i++){
					z = image[x_image + y_image * image_size + i * image_size * image_size] *
						kernel[ty * kernel_size + tx + j * kernel_size * kernel_size];
					atomicAdd(&res[by * res_dim + bx + i * res_dim * res_dim + j * res_dim * res_dim * image_ch], z);
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
void convolution(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int image_ch, int kernel_ch) {
    if(pad > (kernel_size-1)/2){
        std::cout << "Pad is too high" << std::endl;
        return;
    }

    int res_dim = (image_size-kernel_size+2*pad)/stride+1;
    cudaMemset(res, 0, res_dim * res_dim * kernel_ch * sizeof(float));

    convolution_CUDA<<<dim3(res_dim, res_dim), dim3(kernel_size, kernel_size)>>>(image, kernel, res, image_size, kernel_size, stride, pad, res_dim, image_ch, kernel_ch);


    float *ress = new float[res_dim*res_dim*kernel_ch];
    cudaMemcpy(ress, res, res_dim*res_dim*kernel_ch * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] ress;
}

void convolution_weights(float *image, float *kernel, float *res, int image_size, int kernel_size, int stride, int pad, int image_ch, int num_kern) {
	if(pad > (kernel_size-1)/2){
		std::cout << "Pad is too high" << std::endl;
		return;
	}

	int res_dim = (image_size-(stride * (kernel_size - 1) + 1)+2*pad)+1;
	cudaMemset(res, 0, res_dim * res_dim * num_kern * image_ch * sizeof(float));
	convolution_weights_CUDA<<<dim3(res_dim, res_dim), dim3(kernel_size, kernel_size)>>>
	(image, kernel, res, image_size, kernel_size, stride, pad, res_dim, image_ch, num_kern);
}

__global__ void convolution_prevlayer_backpropagation_CUDA(float *cost, float *kernel, float *res, int cost_size, int kernel_size, int edge, int res_dim, int prevlayer_ch, int kernel_ch) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int kernel_len = kernel_size * kernel_size ;
    if (tx * kernel_size + ty < kernel_len) {
        int kernel_left = by - (kernel_size-1)/2;
        int kernel_up = bx - (kernel_size-1)/2;

        float x;
        if ((kernel_left + ty < edge) || //padding a sinistra
            (kernel_left + ty >= cost_size + edge) || //padding a destra
            (kernel_up + tx < edge) || //padding sopra
            (kernel_up + tx >= cost_size + edge)) //padding sotto
            return;


        else {
            int offset = (bx + tx - (kernel_size - 1)/2 - edge) * cost_size + by + ty -
                         (kernel_size - 1) / 2 - edge; // offset da aggiungere  o sottrarre

            int cost_len = cost_size * cost_size;
            int res_len = res_dim * res_dim;
            for (int i = 0; i < prevlayer_ch; i++) {
                for (int j = 0; j < kernel_ch; j++) {
                    x = cost[offset + j * cost_len] *
                        kernel[ kernel_len - 1 - (tx * kernel_size + ty) + j * kernel_len * prevlayer_ch +
                               i * kernel_len];
                    atomicAdd(&res[bx * res_dim + by + i * res_len], x);
                }
            }
        }
    }
}


void convolution_prevlayer_backpropagation(float *cost, float *kernel, float *res, int cost_size, int kernel_size, int prevlayer_size, int kernel_ch, int prevlayer_ch){
    cudaMemset(res, 0, prevlayer_size * prevlayer_size * prevlayer_ch * sizeof(float));
    int edge = (prevlayer_size - cost_size)/2;
    convolution_prevlayer_backpropagation_CUDA<<<dim3(prevlayer_size, prevlayer_size), dim3(kernel_size, kernel_size)>>>(
			cost,
			kernel,
			res,
			cost_size,
			kernel_size,
			edge,
			prevlayer_size,
			prevlayer_ch,
			kernel_ch);


}

//region cpu

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

//endregion