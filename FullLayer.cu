#include "FullLayer.h"

FullLayer::FullLayer(int n_neurons, int linked_neurons, bool isReLU) {
    /**
     * weights: number of rows = weights_len, number of columns = num_neurons
     **/
	this->num_neurons = n_neurons;
	this->weights_len = linked_neurons;
	this->num_weights = n_neurons*linked_neurons;
	this->weights = new float[num_weights];
	this->weights_derivative = new float[num_weights];
    this->activations = new float[n_neurons];
	std::random_device generator;
	std::normal_distribution<float> weights_rand = std::normal_distribution<float>(0.0f, 1.f);
	if(isReLU) {
		activation_func = reLU;
		derivative_func = Heaviside;
	}
	else {
		activation_func = sigmoid;
		derivative_func = der_sigmoid;
	}
	for (int i=0; i<n_neurons*linked_neurons; i++){
		weights[i] = 1.f;
		weights_derivative[i] = 0.0f;
	}
    this->bias = new float[n_neurons];
    this->bias_derivative = new float[n_neurons];
	for (int i=0; i<n_neurons; i++) {
		bias[i] = 0.0f;
		bias_derivative[i] = 0.0f;
	}
}

FullLayer::~FullLayer(){
	delete[] this->weights;
	delete[] this->bias;
    delete[] this->activations;
    delete[] this->weights_derivative;
    delete[] this->bias_derivative;
}

float FullLayer::reLU(float f){
	return f > 0.0f ? f : 0.0f;
}

float FullLayer::sigmoid(float f){
	return 1.f/ (1.f + exp(-f));
}

float FullLayer::der_sigmoid(float f){
	return f*(1 - f);
}

float* FullLayer::forward(float *values) {
	float *val =matrix_mul(values, this->weights, 1, this->weights_len, this->num_neurons);
	//bias sum
	for(int i=0; i<num_neurons; i++){
		val[i] += bias[i];
		val[i] = activation_func(val[i]);
        this->activations[i] = val[i];
	}
	//TODO: memory leakage in val
	return val;
}

int FullLayer::getNeurons() {
    return this->num_neurons;
}

float* FullLayer::getActivations() {
    return this->activations;
}

float FullLayer::Heaviside(float f){
    return f > 0.0f ? 1.0f : 0.0f;
}


float* FullLayer::backpropagation(float* cost, float* back_neurons) {
    // other derivatives are obtained in the same way as the bias derivative but using more terms
    // so we start computing bias derivatives and then use those as baseline for other derivatives
	float* tmp_bias = new float[this->num_neurons];
    for(int i = 0; i < this->num_neurons; i++){
        tmp_bias[i] = derivative_func(this->activations[i])*cost[i];
		bias_derivative[i] += tmp_bias[i];
    }
	delete[] cost;
//    float* res = matrix_mul(back_neurons, tmp_bias, this->weights_len, 1, this->num_neurons);
	float* res = matrix_mulCPU(back_neurons, tmp_bias, this->weights_len, 1, this->num_neurons);
    float* prev_layer_derivative = matrix_mulCPU(tmp_bias, this->weights, 1, this->num_neurons, this->weights_len);

	delete[] tmp_bias;
	//TODO: Da fare in CUDA
	for (int i=0; i<num_weights; i++){
		weights_derivative[i] += res[i];
	}
	delete[] res;

	return prev_layer_derivative;
}

void FullLayer::applyGradient(float lr) {
	for (int i=0; i<num_weights; i++){
		weights[i] -= weights_derivative[i] * lr;
		weights_derivative[i] = 0;
	}
	for(int i = 0; i < this->num_neurons; i++){
		bias[i] -= bias_derivative[i] * lr;
		bias_derivative[i] = 0;
	}
}
