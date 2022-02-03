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
	static float reLU(float f);
	shared_ptr<float[]> backpropagation(shared_ptr<float[]> cost, shared_ptr<float[]> back_neurons);
private:
    int num_neurons;
    int weights_len;
    float *activations;
	float *weights;
	float *bias;
};


#endif //ALEXNET_FULLLAYER_H
