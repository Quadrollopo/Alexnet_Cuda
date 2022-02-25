#include "network.cuh"


Network::Network(int n_input, float lr) {
	input_size = n_input;
	this->lr = lr;
	channels_init = NULL;
}


Network::Network(int img_size, int channel, float lr) {
	input_size = img_size;
	this->lr = lr;
	this->channels_init = channel;
}

Network* Network::addFullLayer(int neurons, Act func){
	int back_neurons;
	if (layers.empty()) {
		back_neurons = input_size;
	}
	else{
		back_neurons = layers.back()->getNeurons();
	}

	layers.push_back(new FullLayer(neurons, back_neurons, func));
	lastLayerType = full;
	return this;
}

Network* Network::addConvLayer(int kern_size, int num_kernels, int stride, bool pad, Act func) {
	int input_conv;
	int channels;
	if (layers.empty()) {
		if(this->channels_init == NULL){
			cout << "Bad network channels initialization: no channels specified" << std::endl;
			exit(-1);
		}
		input_conv = input_size;
		channels = this->channels_init;
	}
	else{
		if(lastLayerType == full){
			std::cout << "Cant add a convolutional layer to a full layer" << std::endl;
			exit(-1);
		}
		if(lastLayerType == conv) {
			ConvLayer *conv = (ConvLayer *) layers.back();
			input_conv = conv->getOutputSize();
			channels = conv->getOutputChannel();
		}
		else {
			PoolingLayer *conv = (PoolingLayer *) layers.back();
			input_conv = conv->getOutputSize();
			channels = conv->getOutputChannel();
		}

	}

	layers.push_back(
			new ConvLayer(input_conv, channels, kern_size, num_kernels, stride, pad, func));
	lastLayerType = conv;
	return this;
}

Network* Network::addPoolLayer(int pool_size, int stride){
	if(lastLayerType != conv){
		cout << "cant add a pool layer to a non conv layer" << endl;
		exit(-1);
	}

	auto* conv = (ConvLayer*)layers.back();
	int input_conv = conv->getOutputSize();
	int channels = conv->getOutputChannel();
	layers.push_back(
			new PoolingLayer(input_conv, channels, pool_size, stride, pool));
	lastLayerType = maxPool;
	return this;
}

float* Network::forward(float input[]) {
	for (Layer *f : layers){
        float *new_input = f->forward(input);
		input = new_input;
	}
	return input;
}


void Network::train(const float output[], const float expected[], float input[]) {
	//Define loss
	float* cost;
	cudaMalloc(&cost, getOutputSize() * sizeof(float));
	loss_cross_entropy_der(output, expected, cost, getOutputSize());
//	vector_diff_alloc(output, expected, cost, getOutputSize());
//	vector_constant_mul(cost,2,getOutputSize());
	float *tmp = cost;
	for(int i=layers.size()-1; i>0; i--){
		tmp = layers[i]->backpropagation(tmp, layers[i-1]->getActivations());
	}
	layers[0]->backpropagation(tmp, input);

	cudaFree(cost);

}

void Network::decreaseLR(){
	lr /= 10;
}

int Network::getOutputSize() {
	return layers.back()->getNeurons();
}

void Network::learn() {
	for (Layer *f : layers){
		f->applyGradient(lr);
	}
}

