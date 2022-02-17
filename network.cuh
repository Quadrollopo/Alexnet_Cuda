
#ifndef ALEXNET_NETWORK_CUH
#define ALEXNET_NETWORK_CUH

#include <memory>
#include <vector>
#include <iostream>
#include "FullLayer.cuh"
#include "ConvLayer.cuh"
#include "PoolingLayer.cuh"
using namespace std;

class Network {
	vector<Layer*> layers;
	int input_size;
	int channels_init;
	float lr;
	enum layer_type{full, conv};
	layer_type lastLayerType;
	int getOutputSize();
public:
	Network(int n_input, float lr);
	Network(int img_size, int channel, float lr);
	Network* addFullLayer(int neurons, Act func);
	Network* addConvLayer(int kern_size, int num_kernels, int stride, bool pad, Act func);
	Network* addPoolLayer(int pool_size, int stride);
	float* forward(float input[]);
	void train(const float output[], const float expected[], float input[]);
	void learn();

};


#endif
