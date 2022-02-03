#include "FullLayer.h"
#include "matrixMul.cuh"
#include <memory>

FullLayer::FullLayer(int n_neurons, int linked_neurons) {
    /**
     * weights: number of rows = weights_len, number of columns = num_neurons
     **/
	this->num_neurons = n_neurons;
	this->weights_len = linked_neurons;
	this->weights = new float[n_neurons*linked_neurons];
	for (int i=0; i<n_neurons*linked_neurons; i++){
		weights[i] = 1.0f;
	}
    this->bias = new float[n_neurons];
	for (int i=0; i<n_neurons; i++) {
		bias[i] = 0.0f;
	}
}

FullLayer::~FullLayer(){
	delete[] this->weights;
	delete[] this->bias;
}

float* FullLayer::forward(float *values) {
    return matrix_mul(values, this->weights,this->bias, this->weights_len, this->num_neurons);
}

int FullLayer::getNeurons() {
    return this->num_neurons;
}