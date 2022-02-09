#include "ConvolutionalLayer.cuh"
#include "utils.cuh"
#include "CUDA/convolution.cuh"
#include "CUDA/vectorSum.cuh"
#include <random>
#include <stdexcept>

ConvolutionalLayer::ConvolutionalLayer(int input_size, int input_channel, int kernel_size, int kernel_channel, int stride, int pad) {
    this->input_size = input_size;
    this->input_channel = input_channel;
    this->kernel_size = kernel_size;
    this->kernel_channel = kernel_channel;
    this->stride = stride;
    this->pad = pad;
    this->output_size = (input_size-kernel_size+2*pad)/stride+1;
    this->output_channel = kernel_channel;
    this->kernel = new float[kernel_size*kernel_size*kernel_channel]();
}

ConvolutionalLayer::~ConvolutionalLayer(){
    delete[] this->kernel;
}

float* ConvolutionalLayer::forward(float *image) {
    auto res = convolution(image, kernel, input_size, kernel_size, stride, pad, input_channel, kernel_channel);
    for(int i = 0; i < output_size * output_size * output_channel; i++)
        res[i] = reLU(res[i]);
    return res;
}

int ConvolutionalLayer::getInputSize() {
    return this->input_size;
}
int ConvolutionalLayer::getInputChannel() {
    return this->input_channel;
}
int ConvolutionalLayer::getKernelSize() {
    return this->kernel_size;
}
int ConvolutionalLayer::getKernelChannel() {
    return this->kernel_channel;
}
int ConvolutionalLayer::getOutputSize() {
    return this->output_size;
}
int ConvolutionalLayer::getOutputChannel() {
    return this->output_channel;
}

float* ConvolutionalLayer::backpropagation() {
    return nullptr;
}

void ConvolutionalLayer::applyGradient() {
}

