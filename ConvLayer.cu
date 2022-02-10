#include "ConvLayer.cuh"
#include "utils.cuh"
#include "CUDA/convolution.cuh"
#include "CUDA/vectorSum.cuh"
#include <random>
#include <stdexcept>

ConvLayer::ConvLayer(int input_size, int input_channel, int kernel_size, int kernel_channel, int stride, int pad)
		: Layer(n_neurons, linked_neurons, func){
    this->input_size = input_size;
    this->input_channel = input_channel;
    this->kernel_size = kernel_size;
    this->kernel_channel = kernel_channel;
    this->stride = stride;
    this->pad = pad;
    this->output_size = (input_size-kernel_size+2*pad)/stride+1;
    this->output_channel = kernel_channel;
    this->kernel = new float[kernel_size*kernel_size*input_channel*kernel_channel]();
}

ConvLayer::~ConvLayer(){
    delete[] this->kernel;
}

float* ConvLayer::forward(float *image) {
    auto res = convolution(image,
                           this->kernel,
                           this->input_size,
                           this->kernel_size,
                           this->stride,
                           this->pad,
                           this->input_channel,
                           this->kernel_channel);
    for(int i = 0; i < output_size * output_size * output_channel; i++)
        res[i] = reLU(res[i]);
    return res;
}

int ConvLayer::getInputSize() {
    return this->input_size;
}
int ConvLayer::getInputChannel() {
    return this->input_channel;
}
int ConvLayer::getKernelSize() {
    return this->kernel_size;
}
int ConvLayer::getKernelChannel() {
    return this->kernel_channel;
}
int ConvLayer::getOutputSize() {
    return this->output_size;
}
int ConvLayer::getOutputChannel() {
    return this->output_channel;
}

void ConvLayer::applyGradient(float lr) {

}

float *ConvLayer::backpropagation(float *cost, float *back_neurons) {
	return nullptr;
}
