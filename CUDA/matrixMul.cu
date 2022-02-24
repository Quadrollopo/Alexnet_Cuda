// System includes
// CUDA runtime
#include "matrixMul.cuh"

__global__ void matrix_mul_CUDA(float *a, float *b, float *c, int b_row, int b_col) {

    // Block index
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    // Thread index
    unsigned int tx = threadIdx.x;
    if(tx < b_row) {
        int tmpa = (int)by/b_col;
        int tmpb = (int)by/b_col;
        float x = a[bx*b_row+1024*tmpa+tx] * b[tx*b_col+1024*b_col*tmpb+by%b_col];
        __syncthreads();

        atomicAdd(&c[bx*b_col+by%b_col], x);
    }

}
void matrix_mul(float *a, float *b, float *c, int a_row, int b_row, int b_col) {
    cudaMemset(c,0,a_row * b_col * sizeof(float));
    int N = 1;
    int t = b_row;
    if(b_row > 1024) {
        N = b_row/1024;
        t = 1024;
    }
    matrix_mul_CUDA<<<dim3(a_row, b_col * N), t>>>(a, b, c, b_row, b_col);

    float * res = new float[a_row * b_col];
    cudaMemcpy(res, c, a_row * b_col * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] res;

}

__global__ void matrix_mul_CUDA2(float *a, float *b, float *c, int b_row, const int b_col) {

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

void matrix_mul2(float *a, float *b, float *c, int a_row, int b_row, int b_col) {
    cudaMemset(c,0,a_row * b_col * sizeof(float));
	matrix_mul_CUDA2<<<1, a_row*b_col>>>(a, b, c, b_row, b_col);
}



/**
 * @param a first matrix (1 x weights_row)
 * @param b second matrix (weights_row x weights_col as array)
 * @param a_row rows of the first matrix
 * @param b_row rows of the second matrix
 * @param b_col column of the second matrix
 * float *values, float *weights, int weights_row, int weights_col
 */


__global__ void matrix_mul_CUDA_shared(float* a, float* b, float* c, int b_row, int b_col){
// Block index
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    // Thread index
    unsigned int tx = threadIdx.x;
    extern __shared__ float s[];
    int t=b_row;
    if(b_row > 1024)
        t = 1024;
    float *s1 = s;
    float *s2 = s1+(t * sizeof(float));
    int tmpa = (int)by/b_col;
    int tmpb = (int)by/b_col;
    s1[tx] = a[bx*b_row+1024*tmpa+tx];
    s2[tx] = b[tx*b_col+1024*b_col*tmpb+by%b_col];
    __syncthreads();
    if(tx < t) {

        float x = s1[tx] * s2[tx];
        atomicAdd(&c[bx*b_col+by%b_col], x);
    }
}

void matrix_mul3(float *a, float *b, float *c, int a_row, int b_row, int b_col) {
    cudaMemset(c,0,a_row * b_col * sizeof(float));
    int N = 1;
    int t = b_row;
    if(b_row > 1024) {
        N = b_row/1024;
        t = 1024;
    }
    matrix_mul_CUDA_shared<<< dim3(a_row, b_col * N), t,2 * t * sizeof(float) >>>(a, b, c, b_row, b_col);
}

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

__global__ void matrix_mul_CUDA4(float *a, float *b, float *c, int b_row, int b_col) {

    // Block index
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    // Thread index
    unsigned int tx = threadIdx.x;
    if(tx < b_row) {
        int tmpa = (int)by/b_col;
        int tmpb = (int)by/b_col;
        float x = a[bx*b_row+1024*tmpa+tx] * b[tx*b_col+1024*b_col*tmpb+by%b_col];
        __syncthreads();

        atomicAdd(&c[bx*b_col+by%b_col], x);
    }

}
void matrix_mul_squared(float *a, float *b, float *c, int N) {
    cudaMemset(c,0,N * N * sizeof(float));
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    if (N*N > 1024){
        threadsPerBlock.x = 1024;
        threadsPerBlock.y = 1024;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }

}


