#include <iostream>
#include "network.cuh"
#include "convolution.cuh"
#include <cmath>
#include <random>

using namespace std;

#define BATCH_SIZE 4
#define NUM_EPOCHS 300

int main() {
	random_device r;
	uniform_int_distribution<int> distribution = uniform_int_distribution<int>(0, 1);
    Network net(2, 2.0f);
    net.addFullLayer(2);
    net.addFullLayer(1);
	float in[2][4] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
	float exp[4] = {0.0f, 1.0f, 1.0f, 0.0f};
	float* out;
	for (int j=0; j < NUM_EPOCHS; j++) {
		double loss = 0.0;
		for (int i = 0; i < BATCH_SIZE; i++) {
			float a[2] = {in[0][i] , in[1][i]};
			out = net.forward(a);
			net.train(out, &exp[i], a);
			loss += pow((out[0] - exp[i]), 2);
            delete[] out;
		}
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
    auto image = new float[image_size*image_size];
    auto kernel = new float[kernel_size*kernel_size];
    auto res_dim = (image_size-kernel_size+2*pad)/stride+1;
    for(int i=0;i<image_size*image_size;i++)
        image[i]=(float)i+1;
    for(int i=0;i<kernel_size*kernel_size;i++)
        kernel[i]=(float)i+1;

    float* res_CUDA = convolution(image,kernel,image_size,kernel_size,stride,pad);
    auto res_CPU = convolution_CPU(image,kernel,kernel_size,image_size,stride,true);
    delete[] res_CUDA;
    delete[] res_CPU;
    return 0;
}
