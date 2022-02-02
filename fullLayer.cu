#include "fullLayer.h"


fullLayer::fullLayer(int n_neurons, int n_weights, fullLayer* next) {
	this->num_neurons = n_neurons;
	this->weights_len = n_weights;
	this->neurons = new float[n_neurons];
	this->nextLayer = next;
	if(next != nullptr){
		weigths = new float* [num];
		for (int i = 0; i < num; ++i)
			weigths[i] = new float [nextLayer->num_neurons];
	}
}

std::shared_ptr<float[]> fullLayer::forward(std::shared_ptr<float[]> values) {
	return values;
}