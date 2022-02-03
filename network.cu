#include "network.h"
#include "FullLayer.h"
#include <cmath>

void Network::addFullLayer(int neurons){
	int back_neurons;
	if (layers.empty()) {
		back_neurons = input_size;
	}
	else{
		back_neurons = layers.back()->getNeurons();
	}

	layers.push_back(new FullLayer(neurons, back_neurons));
}

Network::Network(int n_input) {
	input_size = n_input;
}

float* Network::forward(float input[]) {
	for (FullLayer *f : layers){
		input = f->forward(input);
	}
	return input;
}

void Network::learn(float output[], float expected[]) {
	//Define loss
	float* cost = new float[getOutputSize()];
	for(int i=0; i<getOutputSize(); i++)
		cost[i] = (output[i] - expected[i]) * 2;
	for(int i=layers.size()-1; i>0; i--){
		cost = layers[i]->backpropagation(cost, layers[i-1]->getActivations());
	}
	delete[] cost;
}

int Network::getOutputSize() {
	return layers.back()->getNeurons();
}
