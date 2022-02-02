#ifndef ALEXNET_FULLLAYER_H
#define ALEXNET_FULLLAYER_H
#include <memory>
#include <vector>

class fullLayer {
	int num_neurons;
    int weights_len;
	std::unique_ptr<fullLayer> nextLayer;
    float **neurons;
public:
	fullLayer(int n_neurons, int n_weights, fullLayer* next);
    ~fullLayer();
	std::shared_ptr<float[]> forward(std::shared_ptr<float[]> values);
};


#endif //ALEXNET_FULLLAYER_H
