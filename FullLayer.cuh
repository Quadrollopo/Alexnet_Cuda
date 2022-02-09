#ifndef ALEXNET_FULLLAYER_CUH
#define ALEXNET_FULLLAYER_CUH

#include "CUDA/matrixMul.cuh"
#include "CUDA/vectorSum.cuh"
#include "utils.cuh"
#include "Layer.cuh"
#include <memory>
#include <random>
#include <stdexcept>

using namespace std;

class FullLayer : public Layer{
public:
	FullLayer(int n_neurons, int linked_neurons, Act func);
	~FullLayer();
	float* forward(float *values) override;
	float* backpropagation(float* cost, float* back_neurons) override;
	void applyGradient(float lr) override;
private:
	float *bias;
	float* bias_derivative;
};


#endif ALEXNET_FULLLAYER_CUH
