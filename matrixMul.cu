// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>




__global__ void matrixMul(int *a, int *b, int *c, int rc, int col) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    if(tx<rc) {
        int x = a[bx*rc+tx] * b[tx*col+by];

        __syncthreads();

        atomicAdd(&c[bx*col+by], x);
    }

}
/**
 *
 * @param m1 first matrix
 * @param m2 second matrix
 * @param rows_1 rows of the first matrix
 * @param col_2 column of the second matrix
 * @param r_c rows of second matrix and columns of first matrix
 * @return result matrix
 * float **m1, float **m2, int rows_1, int col_2, int r_c
 */
void matrix_mul() {
//	rows_1 = 4;
//	col_2 = 2;
//	r_c = 3;
    int a1[12] = {7, 1, 3, 1, 9, 2, 5, 4, 2, 6, 9, 3};
    int b1[6] = {1, 5, 5, 7, 3, 4};
    int c1[8] = {0,0,0,0,0,0,0,0};
    int *d_a, *d_b, *d_c;
    /*cudaMalloc(&d_a, 4 * sizeof(int *));

    for (int i = 0; i < 3; i++) {
        cudaMalloc(&d_a[i], 3 * sizeof(int));
        cudaMemcpy(d_a[i], a1, 3 * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&d_b, 3 * sizeof(int *));
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_b[i], 2 * sizeof(int));
        cudaMemcpy(d_b[i], b1, 6 * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&d_c, 4 * sizeof(int *));
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_c[i], 2 * sizeof(int));
        cudaMemcpy(d_c[i]
                , c1, 8 * sizeof(int), cudaMemcpyHostToDevice);
    }
*/
    cudaMalloc(&d_a, 12 * sizeof(int));
    cudaMalloc(&d_b, 6 * sizeof(int));
    cudaMalloc(&d_c, 8 * sizeof(int));
    cudaMemcpy(d_a, a1, 12 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b1, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c1, 8 * sizeof(int), cudaMemcpyHostToDevice);

    matrixMul<<<dim3(4,2), 3>>>(d_a,d_b,d_c,3,2);

    cudaMemcpy(&c1, d_c, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<8;i++)
        printf("%d ",c1[i]);
    cudaDeviceReset();
}
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
//template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
//    float *B, int wA,
//    int wB) {
//  // Block index
//  int bx = blockIdx.x;
//  int by = blockIdx.y;
//
//  // Thread index
//  int tx = threadIdx.x;
//  int ty = threadIdx.y;
//
//  // Index of the first sub-matrix of A processed by the block
//  int aBegin = wA * BLOCK_SIZE * by;
//
//  // Index of the last sub-matrix of A processed by the block
//  int aEnd   = aBegin + wA - 1;
//
//  // Step size used to iterate through the sub-matrices of A
//  int aStep  = BLOCK_SIZE;
//
//  // Index of the first sub-matrix of B processed by the block
//  int bBegin = BLOCK_SIZE * bx;
//
//  // Step size used to iterate through the sub-matrices of B
//  int bStep  = BLOCK_SIZE * wB;
//
//  // Csub is used to store the element of the block sub-matrix
//  // that is computed by the thread
//  float Csub = 0;
//
//  // Loop over all the sub-matrices of A and B
//  // required to compute the block sub-matrix
//  for (int a = aBegin, b = bBegin;
//       a <= aEnd;
//       a += aStep, b += bStep) {
//    // Declaration of the shared memory array As used to
//    // store the sub-matrix of A
//    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
//
//    // Declaration of the shared memory array Bs used to
//    // store the sub-matrix of B
//    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
//
//    // Load the matrices from device memory
//    // to shared memory; each thread loads
//    // one element of each matrix
//    As[ty][tx] = A[a + wA * ty + tx];
//    Bs[ty][tx] = B[b + wB * ty + tx];
//
//    // Synchronize to make sure the matrices are loaded
//    __syncthreads();
//
//    // Multiply the two matrices together;
//    // each thread computes one element
//    // of the block sub-matrix
//#pragma unroll
//
//    for (int k = 0; k < BLOCK_SIZE; ++k) {
//      Csub += As[ty][k] * Bs[k][tx];
//    }
//
//    // Synchronize to make sure that the preceding
//    // computation is done before loading two new
//    // sub-matrices of A and B in the next iteration
//    __syncthreads();
//  }
//
//  // Write the block sub-matrix to device memory;
//  // each thread writes one element
//  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
//  C[c + wB * ty + tx] = Csub;
//}
//
//void ConstantInit(float *data, int size, float val) {
//  for (int i = 0; i < size; ++i) {
//    data[i] = val;
//  }
//}
//
///**
// * Run a simple test of matrix multiplication using CUDA
// */
//int MatrixMultiply(int block_size, const dim3 &dimsA,
//                   const dim3 &dimsB) {
//  // Allocate host memory for matrices A and B
//  unsigned int size_A = dimsA.x * dimsA.y;
//  unsigned int mem_size_A = sizeof(float) * size_A;
//  float *h_A;
//  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
//  unsigned int size_B = dimsB.x * dimsB.y;
//  unsigned int mem_size_B = sizeof(float) * size_B;
//  float *h_B;
//  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
//  cudaStream_t stream;
//
//  // Initialize host memory
//  const float valB = 0.01f;
//  ConstantInit(h_A, size_A, 1.0f);
//  ConstantInit(h_B, size_B, valB);
//
//  // Allocate device memory
//  float *d_A, *d_B, *d_C;
//
//  // Allocate host matrix C
//  dim3 dimsC(dimsB.x, dimsA.y, 1);
//  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
//  float *h_C;
//  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
//
//  if (h_C == NULL) {
//    fprintf(stderr, "Failed to allocate host matrix C!\n");
//    exit(EXIT_FAILURE);
//  }
//
//  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
//  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
//  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
//  // Allocate CUDA events that we'll use for timing
//  cudaEvent_t start, stop;
//  checkCudaErrors(cudaEventCreate(&start));
//  checkCudaErrors(cudaEventCreate(&stop));
//
//  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//
//  // copy host memory to device
//  checkCudaErrors(
//      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
//  checkCudaErrors(
//      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
//
//  // Setup execution parameters
//  dim3 threads(block_size, block_size);
//  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
//
//  // Create and start timer
//  printf("Computing result using CUDA Kernel...\n");
//
//  // Performs warmup operation using matrixMul CUDA kernel
//  if (block_size == 16) {
//    MatrixMulCUDA<16>
//        <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
//  } else {
//    MatrixMulCUDA<32>
//        <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
//  }
//
//  printf("done\n");
//  checkCudaErrors(cudaStreamSynchronize(stream));
//
//  // Record the start event
//  checkCudaErrors(cudaEventRecord(start, stream));
//
//  // Execute the kernel
//  int nIter = 300;
//
//  for (int j = 0; j < nIter; j++) {
//    if (block_size == 16) {
//      MatrixMulCUDA<16>
//          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
//    } else {
//      MatrixMulCUDA<32>
//          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
//    }
//  }
//
//  // Record the stop event
//  checkCudaErrors(cudaEventRecord(stop, stream));
//
//  // Wait for the stop event to complete
//  checkCudaErrors(cudaEventSynchronize(stop));
//
//  float msecTotal = 0.0f;
//  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
//
//  // Compute and print the performance
//  float msecPerMatrixMul = msecTotal / nIter;
//  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
//                             static_cast<double>(dimsA.y) *
//                             static_cast<double>(dimsB.x);
//  double gigaFlops =
//      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
//  printf(
//      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
//      " WorkgroupSize= %u threads/block\n",
//      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);
//
//  // Copy result from device to host
//  checkCudaErrors(
//      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
//  checkCudaErrors(cudaStreamSynchronize(stream));
//
//  printf("Checking computed result for correctness: ");
//  bool correct = true;
//
//  // test relative error by the formula
//  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
//  double eps = 1.e-6;  // machine zero
//
//  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
//    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
//    double dot_length = dimsA.x;
//    double abs_val = fabs(h_C[i]);
//    double rel_err = abs_err / abs_val / dot_length;
//
//    if (rel_err > eps) {
//      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
//             i, h_C[i], dimsA.x * valB, eps);
//      correct = false;
//    }
//  }
//
//  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
//
//  // Clean up memory
//  checkCudaErrors(cudaFreeHost(h_A));
//  checkCudaErrors(cudaFreeHost(h_B));
//  checkCudaErrors(cudaFreeHost(h_C));
//  checkCudaErrors(cudaFree(d_A));
//  checkCudaErrors(cudaFree(d_B));
//  checkCudaErrors(cudaFree(d_C));
//  checkCudaErrors(cudaEventDestroy(start));
//  checkCudaErrors(cudaEventDestroy(stop));
//  printf(
//      "\nNOTE: The CUDA Samples are not meant for performance "
//      "measurements. Results may vary when GPU Boost is enabled.\n");
//
//  if (correct) {
//    return EXIT_SUCCESS;
//  } else {
//    return EXIT_FAILURE;
//  }
//}
//
