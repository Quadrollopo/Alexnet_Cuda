#include "fullLayer.h"


fullLayer::fullLayer(const int num, fullLayer* next) {
	this->num_neurons = num;
	this->neurons = new float [num];
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