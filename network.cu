#include "network.h"
#include "FullLayer.h"


void Network::addFullLayer(int neurons){
	int back_neurons;
	if (layers.empty()) {
		back_neurons = input_size;
	}
	else{
		back_neurons = layers.back()->getNeurons();
	}
	FullLayer *f = new FullLayer(neurons, back_neurons);

	layers.insert(layers.end(), f);
//	layers.insert(1, FullLayer(neurons, back_neurons));
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
