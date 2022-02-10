
#ifndef ALEXNET_NETWORK_CUH
#define ALEXNET_NETWORK_CUH

#include <memory>
#include <vector>
#include "FullLayer.cuh"
#include "ConvLayer.cuh"

using namespace std;

class Network {
	vector<Layer*> layers;
	int input_size;
	int channels;
	float lr;
	enum layer_type{full, conv};
	layer_type lastLayerType;
	int getOutputSize();
public:
	Network(int n_input, float lr);
	Network(int img_size, int channel, float lr);
	void addFullLayer(int neurons, Act func);
	void addConvLayer(int kern_size, int num_kernels, int stride, bool pad, Act func);
	float* forward(float input[]);
	void train(const float output[], const float expected[], float input[]);
	void learn();
};


#endif
