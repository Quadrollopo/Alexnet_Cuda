#ifndef ALEXNET_POOLING_CUH
#define ALEXNET_POOLING_CUH

#include "Layer.cuh"
using namespace std;

class PoolingLayer: public Layer{
public:
    PoolingLayer(int input_size, int input_channel, int kernel_size, int stride, Act func);
    ~PoolingLayer();
    int getInputSize();
    int getInputChannel();
    int getPoolSize();
    int getOutputSize();
    int getOutputChannel();
    float *forward(float *image) override;
    float* backpropagation(float* cost, float* back_neurons) override;
    void applyGradient(float lr) override;
    int getNeurons() override;
    int getNumBackNeurons() override;
private:
    int input_size;
    int input_channel;
    int pool_size;
    int stride;
    int output_size;
    int output_channel;
    int output_len;
    int *max_indexes;
    float *unpooling;
};


#endif
