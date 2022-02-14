#include "network.cuh"


Network::Network(int n_input, float lr) {
	input_size = n_input;
	this->lr = lr;
	channels = NULL;
}


Network::Network(int img_size, int channel, float lr) {
	input_size = img_size;
	this->lr = lr;
	this->channels = channel;
}

void Network::addFullLayer(int neurons, Act func){
	int back_neurons;
	if (layers.empty()) {
		back_neurons = input_size;
	}
	else{
		back_neurons = layers.back()->getNeurons();
	}

	layers.push_back(new FullLayer(neurons, back_neurons, func));
	lastLayerType = full;
}

void Network::addConvLayer(int kern_size, int num_kernels, int stride, bool pad, Act func) {
	int input_conv;
	int channels;
	if (layers.empty()) {
		if(this->channels == NULL){
			std::cout << "Bad network channels initialization: no channels specified" << std::endl;
			exit(-1);
		}
		input_conv = input_size;
		channels = this->channels;
	}
	else{
		if(lastLayerType == full){
			std::cout << "Cant add a convolutional layer to a full layer" << std::endl;
			exit(-1);
		}
		ConvLayer* conv = (ConvLayer*)layers.back();
		input_conv = conv->getOutputSize();
		channels = conv->getOutputChannel();
	}

	layers.push_back(
			new ConvLayer(input_conv, channels, kern_size, num_kernels, stride, pad, func));
	lastLayerType = conv;
}

float* Network::forward(float input[]) {
	for (Layer *f : layers){
        float *new_input = f->forward(input);
#if CUDA
        //cudaFree(input);
#else
        //delete[] input;
#endif
		input = new_input;
	}
	//input = (input,layers.back()->getNeurons());
	return input;
}


void Network::train(const float output[], const float expected[], float input[]) {
#if CUDA
	//Define loss
    float* cost = vector_diff_alloc(output, expected, getOutputSize());
    vector_constant_mul(cost,2,getOutputSize());
	for(int i=layers.size()-1; i>0; i--){
		cost = layers[i]->backpropagation(cost, layers[i-1]->getActivations());
        /*printf("Cost: \n");
        for(int j=0; j<layers[i]->getNumBackNeurons();j++)
            printf("%f ",cost[j]);
        printf("\n");*/
	}
	cost = layers[0]->backpropagation(cost, input);

    //cudaFree(cost);
#else
    //Define loss
	float* cost = new float[getOutputSize()];
	for(int i=0; i<getOutputSize(); i++)
		cost[i] = (output[i] - expected[i]) * 2;
	for(int i=layers.size()-1; i>0; i--){
		cost = layers[i]->backpropagation(cost, layers[i-1]->getActivations());
        /*printf("Cost: \n");
        for(int j=0; j<layers[i]->getNumBackNeurons();j++)
            printf("%f ",cost[j]);
        printf("\n");*/
	}
	cost = layers[0]->backpropagation(cost, input);

    delete[] cost;
#endif
}

int Network::getOutputSize() {
	return layers.back()->getNeurons();
}

void Network::learn(float batch_size) {

	for (Layer *f : layers){
		f->applyGradient(lr/batch_size);
	}
}

