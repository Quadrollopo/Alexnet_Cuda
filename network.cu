#include "network.h"


void network::addLayer(fullLayer layer_new){
	layers.push_back(layer_new);
}

network::network(int n_input) {
	input_size = n_input;
}

shared_ptr<float[]> network::forward(shared_ptr<float[]> input) {
	for (fullLayer &f : layers){
		input = f.forward(input);
	}
	return input;
}
