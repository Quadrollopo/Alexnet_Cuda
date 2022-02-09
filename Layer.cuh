
#ifndef ALEXNET_LAYER_H
#define ALEXNET_LAYER_H

#include "utils.cuh"

enum Act {reLu, Sigmoid, softmax};

class Layer {
public:
	Layer(int n_neurons, int linked_neurons, Act func);
	~Layer();
	virtual float* forward(float *values);
	virtual float* backpropagation(float* cost, float* back_neurons) = 0;
	virtual void applyGradient(float lr) = 0;
	int getNeurons();
	float* getActivations();
	int getNumBackNeurons() const;
private:
	int num_neurons;
	int num_back_neurons;
protected:
	int num_weights;
	float *activations;
	float *weights;
	float* weights_derivative;
	float (*activation_func)(float);
	float (*derivative_func)(float);
};

#endif //ALEXNET_LAYER_H
