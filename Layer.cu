#include "Layer.cuh"


Layer::Layer(Act func) {

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

Layer::~Layer()= default;

float* Layer::getActivations() {
	return this->activations;
}

float *Layer::forward(float *values) {
	return nullptr;
}
