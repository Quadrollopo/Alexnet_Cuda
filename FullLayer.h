#ifndef ALEXNET_FULLLAYER_H
#define ALEXNET_FULLLAYER_H

class FullLayer {
public:
	FullLayer(int n_neurons, int linked_neurons);
	~FullLayer();
	float* forward(float *values);
    int getNeurons();
	static float reLU(float f);
	float* backpropagation(float cost[]);
private:
    int num_neurons;
    int weights_len;
    float *activations;
	float *weights;
	float *bias;
};


#endif //ALEXNET_FULLLAYER_H
