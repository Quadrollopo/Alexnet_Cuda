#include "network.h"


void Network::addFullLayer(int neurons){
	int back_neurons;
	if (layers.empty()) {
		back_neurons = input_size;
	}
	else{
		back_neurons = layers.back().num_neurons;
	}

	layers.emplace_back(neurons, back_neurons);
}

Network::Network(int n_input) {
	input_size = n_input;
}

float* Network::forward(float input[]) {
	for (FullLayer &f : layers){
		input = f.forward(input);
	}
	return input;
}
