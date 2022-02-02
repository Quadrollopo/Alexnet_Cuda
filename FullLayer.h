#ifndef ALEXNET_FULLLAYER_H
#define ALEXNET_FULLLAYER_H
#include <memory>
#include <vector>

class FullLayer {
public:
	int num_neurons;
	FullLayer(int n_neurons, int linked_neurons);
	~FullLayer();
	float* forward(float values[]);

private:
    int weights_len;
	float **weights;
	float *bias;
};


#endif //ALEXNET_FULLLAYER_H
