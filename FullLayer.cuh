#ifndef ALEXNET_FULLLAYER_CUH
#define ALEXNET_FULLLAYER_CUH
#include "matrixMul.cuh"
#include "vectorSum.cuh"
#include "utils.cuh"
#include <memory>
#include <random>
#include <stdexcept>
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


#endif //ALEXNET_FULLLAYER_CUH
