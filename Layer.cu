#include "Layer.cuh"
Layer::Layer(Act func) {
    switch (func) {
        case reLu:
            this->activation_func = reLU_CUDA;
			this->derivative_func = Heaviside_CUDA;
            break;
        case Sigmoid:
			this->activation_func = sigmoid_CUDA;
			this->derivative_func = der_sigmoid_CUDA;
            break;
        case softmax:
			this->activation_func = Softmax_CUDA;
			this->derivative_func = der_sigmoid_CUDA;
			break;
		case pool: //pool layer dont need a activation function
			break;
    }
}

Layer::~Layer()= default;

float* Layer::getActivations() {
	return this->activations;
}

float *Layer::forward(float *values) {
	return nullptr;
}

