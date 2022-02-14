
#ifndef ALEXNET_LAYER_H
#define ALEXNET_LAYER_H

#include "utils.cuh"
#include "CUDA_or_CPU.cuh"

enum Act {reLu, Sigmoid, softmax, pool};

class Layer {
public:
	Layer(Act func);
	~Layer();
	virtual float* forward(float *values);
	virtual float* backpropagation(float* cost, float* back_neurons) = 0;
	virtual void applyGradient(float lr) = 0;
	virtual int getNeurons() = 0;
	virtual int getNumBackNeurons() = 0;
	float* getActivations();
protected:
	int num_weights;
	float *activations;
	float *weights; //number of rows = weights_len, number of columns = num_neurons
	float *weights_derivative;
	float *bias;
	float *bias_derivative;
#if CUDA
	void (*activation_func)(float*, int);
	void (*derivative_func)(float*, float*, int);
#else
	float (*activation_func)(float);
	float (*derivative_func)(float);
#endif

};

#endif
