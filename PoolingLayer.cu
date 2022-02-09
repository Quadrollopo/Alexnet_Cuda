#include "PoolingLayer.cuh"
#include "utils.cuh"
#include "CUDA/maxPooling.cuh"
#include "CUDA/vectorSum.cuh"
#include <random>
#include <stdexcept>

PoolingLayer::PoolingLayer(int input_size, int input_channel, int pool_size, int stride) {
    this->input_size = input_size;
    this->input_channel = input_channel;
    this->pool_size = pool_size;
    this->stride = stride;
    this->output_size = (input_size-pool_size)/stride+1;
    this->output_channel = input_channel;
}

PoolingLayer::~PoolingLayer(){
}

float* PoolingLayer::forward(float *image) {
    auto res = max_pooling(image,
                           this->input_size,
                           this->pool_size,
                           this->stride,
                           this->input_channel);
    return res;
}

int PoolingLayer::getInputSize() {
    return this->input_size;
}
int PoolingLayer::getInputChannel() {
    return this->input_channel;
}
int PoolingLayer::getPoolSize() {
    return this->pool_size;
}
int PoolingLayer::getOutputSize() {
    return this->output_size;
}
int PoolingLayer::getOutputChannel() {
    return this->output_channel;
}
