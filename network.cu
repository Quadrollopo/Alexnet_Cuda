#include "network.cuh"
#include "FullLayer.cuh"
#include <cmath>

void Network::addFullLayer(int neurons, bool relu){
	int back_neurons;
	if (layers.empty()) {
		back_neurons = input_size;
	}
	else{
		back_neurons = layers.back()->getNeurons();
	}

	layers.push_back(new FullLayer(neurons, back_neurons, relu));
}

Network::Network(int n_input, float lr) {
	input_size = n_input;
	this->lr = lr;
}

float* Network::forward(float input[]) {
	for (FullLayer *f : layers){
		input = f->forward(input);
	}
	//input = (input,layers.back()->getNeurons());
	return input;
}


void Network::train(float output[], float expected[], float input[]) {
	//Define loss
	float* cost = new float[getOutputSize()];
	for(int i=0; i<getOutputSize(); i++)
		cost[i] = (output[i] - expected[i]) * 2;
	for(int i=layers.size()-1; i>0; i--){
		cost = layers[i]->backpropagation(cost, layers[i-1]->getActivations());
	}
	cost = layers[0]->backpropagation(cost, input);
	delete[] cost;
}

int Network::getOutputSize() {
	return layers.back()->getNeurons();
}

void Network::learn() {
	for (FullLayer *f : layers){
		f->applyGradient(lr);
	}
}
