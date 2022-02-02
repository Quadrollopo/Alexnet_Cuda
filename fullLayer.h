#ifndef ALEXNET_FULLLAYER_H
#define ALEXNET_FULLLAYER_H
#include <memory>
#include <vector>

class fullLayer {
	int num_neurons;
	std::unique_ptr<float> neurons;
	std::unique_ptr<fullLayer> nextLayer;
public:
	fullLayer(int num, fullLayer* next);
	std::shared_ptr<float[]> forward(std::shared_ptr<float[]> values);
};


#endif //ALEXNET_FULLLAYER_H
