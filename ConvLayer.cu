#include "ConvLayer.cuh"


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
	std::uniform_real_distribution<float> weights_rand = std::uniform_real_distribution<float>(0.0f, 1.0f);
	for (int i=0; i<num_weights; i++) {
		weights[i] = weights_rand(generator);
	}
	this->weights_derivative = new float [num_weights];
	this->bias = new float [output_len]();
	this->bias_derivative = new float [output_len];
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
    auto res = convolution(image,
                           this->weights,
                           this->input_size,
                           this->kernel_size,
                           this->stride,
                           this->pad,
                           this->channels,
                           this->kernel_num);
    for(int i = 0; i < output_len; i++) {
		res[i] += bias[i];
		res[i] = activation_func(res[i]);
		activations[i] = res[i];
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

}

float *ConvLayer::backpropagation(float *cost, float *back_neurons) {
	return nullptr;
}

int ConvLayer::getNeurons() {
	return output_len;
}

int ConvLayer::getNumBackNeurons() {
	return input_size*input_size*channels;
}
