#include <iostream>
#include "network.cuh"
#include "convolution.cuh"
#include "maxPooling.cuh"
#include <cmath>
#include <random>

using namespace std;

#define BATCH_SIZE 4
#define NUM_EPOCHS 300

int main() {
    /*Network net(2, 0.8f);
    net.addFullLayer(2, true);
    net.addFullLayer(1, false);
    float in[2][4] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    float sol[4] = {0.0f, 1.0f, 1.0f, 0.0f};
    float* out;
    for (int j=0; j < NUM_EPOCHS; j++) {
        double loss = 0.0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            int x = i;
//            int x = distribution(r);
            float a[2] = {in[0][x] , in[1][x]};
            out = net.forward(a);
            net.train(out, &sol[x], a);

            loss += pow((out[0] - sol[x]), 2);
        }
        delete[] out;
        net.learn();
        cout <<"loss: " << loss / BATCH_SIZE << endl;
    }
	//int hit = 0;
	/*for (int i = 0; i < NUM_TEST; i++) {
//		int x = distribution(r);
		int x = i;
		float a[2] = {in[0][x] , in[1][x]};
		out = net.forward(a);
		if(out[0] > sol[x] - 0.25f && out[0] < sol[x] + 0.25f){
			hit++;
		}
	}
	cout <<"Test: " << (float) hit/ NUM_TEST << endl;
    */
    int image_size = 5;
    int kernel_size = 3;
    int pad = 1;
    int stride = 2;
    int image_ch = 96;
    int kernel_ch = 96;
    auto image = new float[image_size*image_size*image_ch];
    auto kernel = new float[kernel_size*kernel_size*kernel_ch];
    for(int i=0;i<image_size*image_size*image_ch;i++)
        image[i]=(float)i+1;
    for(int i=0;i<kernel_ch;i++)
        for(int j=0; j<kernel_size*kernel_size; j++)
            kernel[i*kernel_size*kernel_size+j]=(float)i;

    //float* conv_CUDA = convolution(image,kernel,image_size,kernel_size,stride,pad,image_ch,kernel_ch);
    float* res_CUDA = max_pooling(image,image_size,3,2,96);
    auto res_CPU = convolution_CPU(image,kernel,kernel_size,image_size,stride,true);
    //delete[] conv_CUDA;
    delete[] res_CUDA;
    delete[] res_CPU;
    return 0;
}
