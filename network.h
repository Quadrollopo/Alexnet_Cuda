
#ifndef ALEXNET_NETWORK_H
#define ALEXNET_NETWORK_H

#include <memory>
#include "fullLayer.h"
#include "main.cu"

class network {
	vector<fullLayer> layers;
	int input_size;
public:
	network(int n_input);
	void addLayer(fullLayer layer_new);
	shared_ptr<float[]> forward(shared_ptr<float[]> input);

};


#endif //ALEXNET_NETWORK_H
