
#ifndef ALEXNET_NETWORK_H
#define ALEXNET_NETWORK_H

#include <memory>
#include <vector>
#include "FullLayer.h"

using namespace std;

class Network {
	vector<FullLayer*> layers;
	int input_size;
	int getOutputSize();
public:
	Network(int n_input);
	void addFullLayer(int neurons);
	float* forward(float input[]);
	void learn(float output[], float expected[]);
};


#endif //ALEXNET_NETWORK_H
