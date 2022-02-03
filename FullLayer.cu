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
	return f > 0.0f ? f : 0.0f;
}

float* FullLayer::forward(float *values) {
	float *val =matrix_mul(values,
						   this->weights,
                           1,
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

float FullLayer::Heaviside(float f){
    return f > 0.0f ? 1.0f : 0.0f;
}

float* FullLayer::backpropagation(float* cost, float* back_neurons) {
    // other derivatives are obtained in the same way as the bias derivative but using more terms
    // so we start computing bias derivatives and then use those as baseline for other derivatives
    auto bias_derivative = new float[this->num_neurons];
    for(int i = 0; i < this->num_neurons; i++){
        bias_derivative[i] = Heaviside(this->activations[i])*cost[i];
    }
    float* weights_derivatives = matrix_mul(bias_derivative, back_neurons, this->num_neurons, 1, this->weights_len);
    float* prev_layer_derivative = matrix_mul(this->weights, bias_derivative, this->weights_len, this->num_neurons, 1);

    delete[] bias_derivative;
    delete[] weights_derivatives;


	return prev_layer_derivative; // CHIAMARE DELETE IN NETWORK
}
