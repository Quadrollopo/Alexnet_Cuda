#include "Layer.cuh"


Layer::Layer(Act func) {
	/**
	 * weights: number of rows = weights_len, number of columns = num_neurons
	 **/
	switch (func) {
		case reLu:
			activation_func = reLU;
			derivative_func = Heaviside;
			break;
		case Sigmoid:
			activation_func = sigmoid;
			derivative_func = der_sigmoid;
			break;
		case softmax:
			activation_func = sigmoid;
			derivative_func = der_sigmoid;
	}
}

Layer::~Layer(){
	delete[] this->weights;
	delete[] this->activations;
	delete[] this->weights_derivative;
}

int Layer::getNeurons() {
	return this->num_neurons;
}

float* Layer::getActivations() {
	return this->activations;
}

int Layer::getNumBackNeurons() const {
	return num_back_neurons;
}

float *Layer::forward(float *values) {
	return nullptr;
}
