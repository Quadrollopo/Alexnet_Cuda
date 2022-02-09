#include "FullLayer.cuh"

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
	std::uniform_real_distribution<float> weights_rand = std::uniform_real_distribution<float>(0.0f, 1.f);
	if(isReLU) {
		activation_func = reLU;
		derivative_func = Heaviside;
	}
	else {
		activation_func = sigmoid;
		derivative_func = der_sigmoid;
	}
	for (int i=0; i<n_neurons*linked_neurons; i++){
		weights[i] = weights_rand(generator);
// 		weights[i] = 1.0f;
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

float* FullLayer::forward(float *values) {
	float *val =matrix_mul(values,
                           this->weights,
                           1,
                           this->weights_len,
                           this->num_neurons);
    //delete[] values;
	//bias sum
	for(int i=0; i<num_neurons; i++){
		val[i] += bias[i];
		val[i] = activation_func(val[i]);
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

void cmp(float *a, float *b, int len){
    for(int i=0; i<len; i++){
        if(a[i] != b[i]) {
            printf("Vettori diversi\n\n");
            return;
        }
    }
}

float* FullLayer::backpropagation(float* cost, float* back_neurons) {
    // other derivatives are obtained in the same way as the bias derivative but using more terms
    // so we start computing bias derivatives and then use those as baseline for other derivatives
	float* current_bias_derivative = new float[this->num_neurons];
    float* weights_derivative_CPU = new float[this->num_weights];
    for(int i = 0; i < this->num_neurons; i++){
        current_bias_derivative[i] = derivative_func(this->activations[i]) * cost[i];
		bias_derivative[i] += current_bias_derivative[i];
    }
	delete[] cost;
	float* current_weights_derivative = matrix_mul_CPU(current_bias_derivative,
                                                       back_neurons,
                                                       this->num_neurons,
                                                       1,
                                                       this->weights_len);
    float* prev_layer_derivative = matrix_mul_CPU(this->weights,
                                                  current_bias_derivative,
                                                  this->weights_len,
                                                  this->num_neurons,
                                                  1);

	delete[] current_bias_derivative;

	for (int i=0; i<num_weights; i++){
		weights_derivative[i] += current_weights_derivative[i];
	}
//    vector_sum(weights_derivative, current_weights_derivative, num_weights);
//    weights_derivative_CPU = vector_sum_CPU(weights_derivative, current_weights_derivative, num_weights);
//    cmp(weights_derivative,weights_derivative_CPU,num_weights);

	delete[] current_weights_derivative;
    delete[] weights_derivative_CPU;
	return prev_layer_derivative;
}

void FullLayer::applyGradient(float lr) {
    //fare in CUDA
	for (int i=0; i<num_weights; i++){
		weights[i] -= weights_derivative[i] * lr;
		weights_derivative[i] = 0;
	}
	for(int i = 0; i < this->num_neurons; i++){
		bias[i] -= bias_derivative[i] * lr;
		bias_derivative[i] = 0;
	}
}

