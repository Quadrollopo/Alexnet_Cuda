#include "FullLayer.cuh"

FullLayer::FullLayer(int n_neurons, int linked_neurons, Act func) : Layer(func){
    /**
     * weights: number of rows = weights_len, number of columns = num_neurons
     **/
	this->num_neurons = n_neurons;
	this->num_back_neurons = linked_neurons;
	this->num_weights = n_neurons*linked_neurons;
	this->weights = new float[num_weights];
	this->weights_derivative = new float[num_weights]();
	this->activations = new float[n_neurons];
	this->bias = new float[n_neurons]();
	this->bias_derivative = new float[n_neurons]();
	std::random_device generator;
	std::uniform_real_distribution<float> weights_rand = std::uniform_real_distribution<float>(0.0f, 1.0f);
	for (int i=0; i<num_weights; i++){
		weights[i] = weights_rand(generator);
//		weights[i] = 1.0f;
	}
}

FullLayer::~FullLayer(){
	Layer::~Layer();
	delete[] this->bias;
    delete[] this->bias_derivative;
	delete[] this->weights;
	delete[] this->activations;
	delete[] this->weights_derivative;
}

float* FullLayer::forward(float *values) {
	float *val =matrix_mul_CPU(values,
                           this->weights,
                           1,
                           this->getNumBackNeurons(),
                           this->getNeurons());
    //delete[] values;
	//bias sum
	for(int i=0; i<getNeurons(); i++){
		val[i] += bias[i];
		val[i] = activation_func(val[i]);
        this->activations[i] = val[i];
	}
	return val;
}

float* FullLayer::backpropagation(float* cost, float* back_neurons) {
    // other derivatives are obtained in the same way as the bias derivative but using more terms
    // so we start computing bias derivatives and then use those as baseline for other derivatives
	float* current_bias_derivative = new float[num_neurons];
    for(int i = 0; i < num_neurons; i++){
        current_bias_derivative[i] = derivative_func(this->activations[i]) * cost[i];
		bias_derivative[i] += current_bias_derivative[i];
    }

	delete[] cost;
	float* current_weights_derivative = matrix_mul_CPU(back_neurons,
                                                       current_bias_derivative,
                                                       this->num_back_neurons,
                                                       1,
                                                       this->num_neurons);
    float* prev_layer_derivative = matrix_mul_CPU(this->weights,
                                                  current_bias_derivative,
                                                  this->getNumBackNeurons(),
                                                  this->getNeurons(),
                                                  1);

	delete[] current_bias_derivative;

	for (int i=0; i<num_weights; i++){
		weights_derivative[i] += current_weights_derivative[i];
	}
//    vector_sum(weights_derivative, current_weights_derivative, num_weights);
//    weights_derivative_CPU = vector_sum_CPU(weights_derivative, current_weights_derivative, num_weights);

	delete[] current_weights_derivative;
	return prev_layer_derivative;
}

void FullLayer::applyGradient(float lr) {
    //fare in CUDA
	for (int i=0; i<num_weights; i++){
		weights[i] -= weights_derivative[i] * lr;
		weights_derivative[i] = 0;
	}
	for(int i = 0; i < this->getNeurons(); i++){
		bias[i] -= bias_derivative[i] * lr;
		bias_derivative[i] = 0;
	}
}

int FullLayer::getNeurons() {
	return num_neurons;
}

int FullLayer::getNumBackNeurons() {
	return num_back_neurons;
}
