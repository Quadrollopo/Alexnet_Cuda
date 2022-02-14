#include "ConvLayer.cuh"
#include <iostream>

ConvLayer::ConvLayer(int input_size, int channels, int kernel_size, int kernel_num,
					 int stride, bool pad, Act func)
		: Layer(func){
    this->input_size = input_size;
    this->channels = channels;
    this->kernel_size = kernel_size;
    this->kernel_num = kernel_num;
    this->stride = stride;
	if(pad)
    	this->pad = (kernel_size - 1) / 2;
	else
		this->pad = 0;
    this->output_size = ((input_size-kernel_size+2*pad)/stride+1);
    this->output_len = output_size*output_size*kernel_num;
	this->num_weights = kernel_size*kernel_size*channels*kernel_num;
	this->weights = new float [num_weights];
	std::random_device generator;
	std::uniform_real_distribution<float> weights_rand = std::uniform_real_distribution<float>(0.0f, 0.01f);
	for (int i=0; i<num_weights; i++) {
		weights[i] = weights_rand(generator);
	}
	this->weights_derivative = new float [num_weights]();
	this->bias = new float [kernel_num]();
	this->bias_derivative = new float [kernel_num]();
	this->activations = new float [output_len];
}

ConvLayer::~ConvLayer(){
	Layer::~Layer();
    delete[] this->weights;
    delete[] this->weights_derivative;
	delete[] this->bias;
	delete[] this->bias_derivative;
}

float* ConvLayer::forward(float *image) {
    auto res = convolution_CPU(image,
                           this->weights,
                           this->input_size,
                           this->kernel_size,
                           this->stride,
                           this->pad,
                           this->channels,
                           this->kernel_num);
	for (int i = 0, w=0; w < kernel_num; w++, i+=output_size*output_size) {
		for (int j=0; j < output_size*output_size; j++) {
			res[i+j] += bias[w];
			res[i+j] = activation_func(res[i+j]);
			activations[i+j] = res[i+j];
		}
	}

    return res;
}

int ConvLayer::getInputSize() {
    return this->input_size;
}
int ConvLayer::getChannel() {
    return this->channels;
}
int ConvLayer::getKernelSize() {
    return this->kernel_size;
}
int ConvLayer::getOutputSize() {
    return this->output_size;
}
int ConvLayer::getOutputChannel() {
    return this->kernel_num;
}

void ConvLayer::applyGradient(float lr) {
	for (int i=0; i<num_weights; i++){
		weights[i] -= weights_derivative[i] * lr;
		weights_derivative[i] = 0;
	}
	for(int i = 0; i < kernel_num; i++){
		bias[i] -= bias_derivative[i] * lr;
		bias_derivative[i] = 0;
	}
}


float *ConvLayer::backpropagation(float *cost, float *back_img) {
	float* der_cost = new float[output_len];
	for(int i = 0; i < this->output_len; i++){
		der_cost[i] = derivative_func(this->activations[i]) * cost[i];
		bias_derivative[i / (output_size*output_size)] += der_cost[i];
	}

	float* current_weights_derivative =
			convolution_weights_CPU(back_img,
									der_cost,
									this->input_size,
									this->output_size,
									this->stride,
									pad,
									channels,
									kernel_num);

	for (int i=0; i<num_weights; i++){
		weights_derivative[i] += current_weights_derivative[i];
	}


	auto prev_layer_derivative = convolution_cost_CPU(der_cost,
											 this->weights,
											 this->output_size,
											 this->kernel_size,
											 this->stride,
											 kernel_size - 1,
											 kernel_num,
											 channels);

	return prev_layer_derivative;
}

int ConvLayer::getNeurons() {
	return output_len;
}

int ConvLayer::getNumBackNeurons() {
	return input_size*input_size*channels;
}
