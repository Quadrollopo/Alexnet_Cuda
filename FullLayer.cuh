#ifndef ALEXNET_FULLLAYER_CUH
#define ALEXNET_FULLLAYER_CUH

#include "CUDA/matrixMul.cuh"
#include "CUDA/vectorSum.cuh"
#include "utils.cuh"
#include "Layer.cuh"
#include <memory>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>
using namespace std;

class FullLayer : public Layer{
public:
	FullLayer(int n_neurons, int linked_neurons, Act func);
	~FullLayer();
	float* forward(float *values) override;
	float* backpropagation(float* cost, float* back_neurons) override;
	void applyGradient(float lr) override;
	int getNeurons() override;
	int getNumBackNeurons() override;
private:
    float *neurons;
	int num_neurons;
	int num_back_neurons;
    float *current_bias_derivative;
    float *current_weights_derivative;
    float *prev_layer_derivative;
};


#endif
