#include "FullLayer.h"

FullLayer::FullLayer(int n_neurons, int linked_neurons) {
    /**
     * weights: number of rows = weights_len, number of columns = num_neurons
     **/
	this->num_neurons = n_neurons;
	this->weights_len = linked_neurons;
	this->weights = new float[n_neurons*linked_neurons];
    this->activations = new float[n_neurons];
	std::random_device generator;
	std::normal_distribution<float> weights_rand = std::normal_distribution<float>(0.0f, 0.1f);
	for (int i=0; i<n_neurons*linked_neurons; i++){
		weights[i] = weights_rand(generator);
	}
    this->bias = new float[n_neurons];
	for (int i=0; i<n_neurons; i++) {
		bias[i] = 1.0f;
	}
}

FullLayer::~FullLayer(){
	delete[] this->weights;
	delete[] this->bias;
    delete[] this->activations;
}

float FullLayer::reLU(float f){
	return f > 0 ? f : 0;
}

float* FullLayer::forward(float *values) {
	float *val =matrix_mul(values,
						   this->weights,
						   this->bias,
						   this->weights_len,
						   this->num_neurons);
	//bias sum
	for(int i=0; i<num_neurons; i++){
		val[i] += bias[i];
		val[i] = reLU(val[i]);
        this->activations[i] = val[i];
	}
	return val;
}

int FullLayer::getNeurons() {
    return this->num_neurons;
}

float* FullLayer::getActivations() {
    return this->activations;
}

shared_ptr<float[]> FullLayer::backpropagation(shared_ptr<float[]> cost, shared_ptr<float[]> back_neurons) {

	return nullptr;
}
