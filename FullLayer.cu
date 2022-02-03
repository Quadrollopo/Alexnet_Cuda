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
    this->bias = new float[n_neurons];
}

FullLayer::~FullLayer(){
	delete[] this->weights;
	delete[] this->bias;
}

void FullLayer::forward(float *values, float *res) {
    matrix_mul(values, this->weights, res, this->weights_len, this->num_neurons);
}

int FullLayer::getNeurons() {
    return this->num_neurons;
}