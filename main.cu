#include <iostream>
#include "network.cuh"
#include "Layer.cuh"
#include "CUDA/matrixMul.cuh"
#include "CUDA/convolution.cuh"
#include "CUDA/maxPooling.cuh"
#include <cmath>
#include <random>
#include "chrono"

using namespace std;

#define BATCH_SIZE 64
#define NUM_EPOCHS 1000
#define NUM_TEST 4

int main() {
    random_device r;
    uniform_int_distribution<int> distribution = uniform_int_distribution<int>(0, 3);
//    float a[50], b[9];
//    for (float & x : a)
//        x = distribution(r);
    Network net(2, 0.5f);
    net.addFullLayer(2, Sigmoid);
    net.addFullLayer(1, Sigmoid);
    float *out;
    float in[2][4] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    float sol[4] = {0.0f, 1.0f, 1.0f, 0.0f};
    float loss = 0.0;
    for (int j=0; j < NUM_EPOCHS; j++) {
        loss = 0.0f;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < BATCH_SIZE; i++) {
            //int x = i;
          int x = distribution(r);
            float *a = new float[2];
            a[0] = in[0][x];
            a[1] = in[1][x];
            //printf("\n\nInput: %f %f\n\n", a[0], a[1]);
            float *aa;
            cudaMalloc(&aa, 2 * sizeof(float));
            cudaMemcpy(aa, a, 2 * sizeof(float), cudaMemcpyHostToDevice);

            //float sol[1] = {(a[0] == 1.f) != (a[1] == 1.f) ? 1.f : 0.f};

            float *soll;
            cudaMalloc(&soll, sizeof(float));
            cudaMemcpy(soll, &sol[x], sizeof(float), cudaMemcpyHostToDevice);


            out = net.forward(aa);
            float *out_host = new float[1];
            cudaMemcpy(out_host, out, sizeof(float), cudaMemcpyDeviceToHost);
            net.train(out, soll, aa);
            cudaFree(aa);

            float *sol_host = new float[1];
            cudaMemcpy(sol_host, soll, sizeof(float), cudaMemcpyDeviceToHost);
//            printf("\nout-sol: %f,\n", (out_host[0] - sol_host[0]) * (out_host[0] - sol_host[0]));
//            printf("\nout: %f, sol:%f\n", out_host[0], sol_host[0]);
            loss += (out_host[0] - sol_host[0]) * (out_host[0] - sol_host[0]);
            delete[] out_host;
            delete[] sol_host;
            cudaFree(soll);
        }
        cudaFree(out);
        net.learn((float)BATCH_SIZE);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        //std::cout << "elapsed time total: " << elapsed_seconds.count() << "s\n";
        cout <<"loss: " << loss/BATCH_SIZE << endl;
    }
//    int hit = 0;
//    for (int i = 0; i < NUM_TEST; i++) {
//        int x = distribution(r);
//        int x = i;
//        float a[2] = {in[0][x] , in[1][x]};
//        float sol[1] = {(a[0] == 1.f) != (a[1] == 1.f) ? 1.f : 0.f};
//        out = net.forward(a);
//        if(abs(out[0] - sol[0]) < 0.25f){
//            hit++;
//        }
//    }
//    cout <<"Test: " << (float) hit/ NUM_TEST << endl;
    int image_size = 200;
    int kernel_size = 3;
    int pad = 1;
    int stride = 2;
    int image_ch = 3;
    int kernel_ch = 10;



//    auto image = new float[image_size*image_size*image_ch];
//    auto kernel = new float[kernel_size*kernel_size*kernel_ch*image_ch];
//    for(int i=0;i<image_size*image_size*image_ch;i++)
//        image[i]=(float)i+1;
//    for(int i=0; i<kernel_ch; i++) {
//        for (int j = 0; j < image_ch; j++){
//             for (int k = 0; k < kernel_size * kernel_size; k++) {
//                kernel[i * kernel_size * kernel_size * image_ch + j * kernel_size * kernel_size + k] =
//                        (float)i * kernel_size * kernel_size * image_ch + j * kernel_size * kernel_size + k + 1;
//                //printf("%.1f ",kernel[i * kernel_size * kernel_size * image_ch + j * image_ch + k]);
//            }
//             //printf("\n");
//        }
//    }
//    float *d_image, *d_kernel;
//    cudaMalloc(&d_image, image_size * image_size * image_ch * sizeof(float));
//    cudaMalloc(&d_kernel, kernel_size * kernel_size * image_ch * kernel_ch * sizeof(float));
//
//    cudaMemcpy(d_image, image, image_size * image_size * image_ch * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * image_ch * kernel_ch * sizeof(float), cudaMemcpyHostToDevice);
//
//    //float* conv_CUDA = convolution(image,kernel,image_size,kernel_size,stride,pad,image_ch,kernel_ch);
//    float* res_CUDA = convolution(d_image,d_kernel,image_size,kernel_size,stride,pad,image_ch,kernel_ch);
//    //auto res_CPU = convolution_CPU(image,kernel,kernel_size,image_size,stride,true);
//    //delete[] conv_CUDA;
//    delete[] res_CUDA;
//    delete[] image;
//    delete[] kernel;
//    //delete[] res_CPU;
//
//    cudaFree(d_image);
//    cudaFree(d_kernel);

//    auto image1 = new float[image_size*image_size];
//    auto image2 = new float[image_size*image_size];
//    for(int i=0;i<image_size*image_size;i++){
//        image1[i]=(float)i+1;
//        image2[i]=(float)i+1;
//    }
//    float *d_image1, *d_image2;
//    cudaMalloc(&d_image1, image_size * image_size * sizeof(float));
//    cudaMalloc(&d_image2, image_size * image_size * sizeof(float));
//
//    cudaMemcpy(d_image1, image1, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_image2, image2, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice);
//
//    float* res_CUDA = matrix_mul(d_image1,d_image2,image_size,image_size,image_size);
//    float* res_CUDA2 = matrix_mul(res_CUDA,d_image2,image_size,image_size,image_size);
//
//    delete[] image1;
//    delete[] image2;
//    //delete[] res_CPU;
//    cudaFree(res_CUDA);
//    cudaFree(res_CUDA2);
//    cudaFree(d_image1);
//    cudaFree(d_image2);
    return 0;
}
