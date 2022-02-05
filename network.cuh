
#ifndef ALEXNET_NETWORK_CUH
#define ALEXNET_NETWORK_CUH

#include <memory>
#include <vector>
#include "FullLayer.cuh"

using namespace std;

class Network {
	vector<FullLayer*> layers;
	int input_size;
	float lr;
	int getOutputSize();
public:
	Network(int n_input, float lr);
	void addFullLayer(int neurons, bool relu = true);
	float* Softmax(float input[], int length);
	float* forward(float input[]);
	void train(float output[], float expected[], float input[]);
	void learn();
};


#endif //ALEXNET_NETWORK_CUH
