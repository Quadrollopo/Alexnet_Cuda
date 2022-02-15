#include "PoolingLayer.cuh"

#include "utils.cuh"
#include "CUDA/maxPooling.cuh"
#include "CUDA/vectorSum.cuh"
#include <random>
#include <stdexcept>

PoolingLayer::PoolingLayer(int input_size, int input_channel, int pool_size, int stride, Act func) : Layer(func) {
    this->input_size = input_size;
    this->input_channel = input_channel;
    this->pool_size = pool_size;
    this->stride = stride;
    this->output_size = (input_size-pool_size)/stride+1;
    this->output_channel = input_channel;
    this->output_len = output_size * output_size * input_channel;
    cudaMalloc(&this->activations, output_len * sizeof(float));
    cudaMalloc(&this->max_indexes, output_len * sizeof(int));
    cudaMalloc(&this->unpooling, input_size * input_size * input_channel * sizeof(float));
}

PoolingLayer::~PoolingLayer(){
    Layer::~Layer();
    cudaFree(this->activations);
    cudaFree(this->max_indexes);
    cudaFree(this->unpooling);
}

float* PoolingLayer::forward(float *image) {
    max_pooling(image,
               activations,
               max_indexes,
               this->input_size,
               this->pool_size,
               this->stride,
               this->input_channel);
    return activations;
}

float* PoolingLayer::backpropagation(float* cost, float* back_neurons){
    max_unpooling(activations,
                  max_indexes,
                  unpooling,
                  output_size, //we are going backward
                  input_size, //input and output dimensions are inverted
                  input_channel);
    return unpooling;
}
void PoolingLayer::applyGradient(float lr){
}
int PoolingLayer::getNeurons(){
    return output_size*output_size*output_channel;
}
int PoolingLayer::getNumBackNeurons(){
    return input_size;
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
