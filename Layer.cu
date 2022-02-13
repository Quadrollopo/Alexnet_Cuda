#include "Layer.cuh"
Layer::Layer(Act func) {
#if CUDA
    switch (func) {
        case reLu:
            activation_func = reLU_CUDA;
            derivative_func = Heaviside_CUDA;
            break;
        case Sigmoid:
            activation_func = sigmoid_CUDA;
            derivative_func = der_sigmoid_CUDA;
            break;
        case softmax:
            activation_func = Softmax_CUDA;
            derivative_func = der_sigmoid_CUDA;
        case last:
            activation_func = nope;
            derivative_func = nope_der;
    }
#else
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
#endif
}

Layer::~Layer()= default;

float* Layer::getActivations() {
	return this->activations;
}

float *Layer::forward(float *values) {
	return nullptr;
}


