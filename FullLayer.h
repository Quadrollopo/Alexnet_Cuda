#ifndef ALEXNET_FULLLAYER_H
#define ALEXNET_FULLLAYER_H
#include "matrixMul.cuh"
#include <memory>
#include <random>
using namespace std;

class FullLayer {
public:
	FullLayer(int n_neurons, int linked_neurons);
	~FullLayer();
	float* forward(float *values);
    int getNeurons();
    float *getActivations();
	static float reLU(float f);
	float* backpropagation(float* cost, float* back_neurons);
    float FullLayer::Heaviside(float f);
	void applyGradient(float lr);
private:
    int num_neurons;
    int weights_len;
    float *activations;
	float *weights;
	float *bias;
	float* bias_derivative;
	float* weights_derivative;
	int num_weights;
};


#endif //ALEXNET_FULLLAYER_H
