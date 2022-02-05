#ifndef ALEXNET_FULLLAYER_H
#define ALEXNET_FULLLAYER_H
#include "matrixMul.cuh"
#include "vectorSum.cuh."
#include <memory>
#include <random>
using namespace std;

class FullLayer {
public:
	FullLayer(int n_neurons, int linked_neurons, bool isReLU = true);
	~FullLayer();
    int getNeurons();
	float* forward(float *values);
    float* getActivations();
	float* backpropagation(float* cost, float* back_neurons);
	void applyGradient(float lr);
private:
	static float reLU(float f);
    static float Heaviside(float f);
	static float sigmoid(float f);
	static float der_sigmoid(float f);
    int num_neurons;
    int weights_len;
    float *activations;
	float *weights;
	float *bias;
	float* bias_derivative;
	float* weights_derivative;
	int num_weights;
	float (*activation_func)(float);
	float (*derivative_func)(float);
};


#endif //ALEXNET_FULLLAYER_H
