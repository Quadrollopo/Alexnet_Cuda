#include "network.cuh"

void Network::addFullLayer(int neurons, Act func){
	int num_back_neurons;
	if (layers.empty()) {
        num_back_neurons = input_size;
	}
	else{
		num_back_neurons = layers.back()->getNeurons();
	}

	layers.push_back(new FullLayer(neurons, num_back_neurons, func));
}

Network::Network(int n_input, float lr) {
	input_size = n_input;
	this->lr = lr;
}

float* Network::forward(float input[]) {
	for (Layer *f : layers){
		input = f->forward(input);
	}
	//input = (input,layers.back()->getNeurons());
	return input;
}


void Network::train(const float output[], const float expected[], float input[]) {
	//Define loss
	float* cost = new float[getOutputSize()];
    //printf("Cost: \n");
	for(int i=0; i<getOutputSize(); i++){
        cost[i] = (output[i] - expected[i]) * 2; //rivedere
        //printf("%f ",cost[i]);
    }

   // printf("\n");
	for(int i=layers.size()-1; i>0; i--){
		cost = layers[i]->backpropagation(cost, layers[i-1]->getActivations());
	}
	cost = layers[0]->backpropagation(cost, input);
	delete[] cost;
}

int Network::getOutputSize() {
	return layers.back()->getNeurons();
}

void Network::learn() {
	for (Layer *f : layers){
		f->applyGradient(lr);
	}
}
