// System includes
#include <stdio.h>

// CUDA runtime
#include "matrixMul.cuh"
#include <cuda_runtime.h>


__global__ void matrix_mul_CUDA2(float *a, float *b, float *c, const int a_row, int b_row, const int b_col) {

	// Thread index
	unsigned int tx = threadIdx.x;
	unsigned int col = tx % b_row;
	unsigned int row = tx - col;
	float sum = 0;
#pragma unroll
	for (int i=0, j=0; i<b_row; i++, j+=b_col){
		sum += a[row+i] * b[j+col];
	}
	__syncthreads();

	c[tx] = sum;

}


__global__ void matrix_mul_CUDA(float *a, float *b, float *c, int a_row, int b_row, int b_col) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    if(tx < b_row) {
        float x = a[bx*b_row+tx] * b[tx*b_col+by];
        __syncthreads();

        atomicAdd(&c[bx*b_col+by], x);
    }

}
/**
 * @param a first matrix (1 x weights_row)
 * @param b second matrix (weights_row x weights_col as array)
 * @param a_row rows of the first matrix
 * @param b_row rows of the second matrix
 * @param b_col column of the second matrix
 * float *values, float *weights, int weights_row, int weights_col
 **/
float* matrix_mul(float *a, float *b, int a_row, int b_row, int b_col) {

    float *d_a, *d_b, *d_c;

    auto res = new float[a_row * b_col]();

    cudaMalloc(&d_a, a_row * b_row * sizeof(float));
    cudaMalloc(&d_b, b_row * b_col * sizeof(float));
    cudaMalloc(&d_c, a_row * b_col * sizeof(float));

    cudaMemcpy(d_a, a, a_row * b_row * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_row * b_col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, res, a_row * b_col * sizeof(float), cudaMemcpyHostToDevice);

    matrix_mul_CUDA<<<dim3(a_row, b_col), b_row>>>(d_a, d_b, d_c, a_row, b_row, b_col);

    cudaMemcpy(res, d_c, a_row * b_col * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

//    for(int i=0; i < a_row * b_col; i++){
//        printf("%f ", res[i]);
//    }
//    printf("\n\n\n\n");


    cudaDeviceReset();

    return res;
}

float* matrix_mul2(float *a, float *b, int a_row, int b_row, int b_col) {

	float *d_a, *d_b, *d_c;

	auto res = new float[a_row * b_col];

	for(int i=0; i < a_row * b_col; i++)
		res[i] = 0.0f;


	cudaMalloc(&d_a, a_row * b_row * sizeof(float));
	cudaMalloc(&d_b, b_row * b_col * sizeof(float));
	cudaMalloc(&d_c, a_row * b_col * sizeof(float));

	cudaMemcpy(d_a, a, a_row * b_row * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, b_row * b_col * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, res, a_row * b_col * sizeof(float), cudaMemcpyHostToDevice);

	matrix_mul_CUDA2<<<1, a_row*b_col, a_row*b_col*sizeof(float)>>>(d_a, d_b, d_c, a_row, b_row, b_col);

	cudaMemcpy(res, d_c, a_row * b_col * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

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
float* matrix_mul_CPU(float *a, float *b, int a_row, int b_row, int b_col) {

	auto res = new float[a_row * b_col];

	for(int i=0; i < a_row * b_col; i++)
		res[i] = 0.0f;

	for(int i = 0; i < a_row; i++)
		for(int j=0; j < b_col; j++)
			for(int k = 0; k < b_row; k++)
				res[i*b_col+j] += a[i*b_row+k] * b[k*b_col+j] ;

	return res;
}
