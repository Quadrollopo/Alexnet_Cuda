#include "FullLayer.h"
#include <memory>

FullLayer::FullLayer(int n_neurons, int linked_neurons) {
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

FullLayer::~FullLayer(){
	for (int i = 0; i < this->num_neurons; ++i)
		delete[] (this->weights[i]);
	delete[] (this->weights);
	delete[] bias;
}

float* FullLayer::forward(float values[]) {
	//TODO: matrix mul
	return values;
}