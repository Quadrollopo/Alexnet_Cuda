#ifndef ALEXNET_FULLLAYER_H
#define ALEXNET_FULLLAYER_H
#include <memory>

class FullLayer {
public:
	FullLayer(int n_neurons, int linked_neurons);
	~FullLayer();
	float* forward(float *values);
    int getNeurons();
private:
    int num_neurons;
    int weights_len;
	float *weights;
	float *bias;
};


#endif //ALEXNET_FULLLAYER_H
