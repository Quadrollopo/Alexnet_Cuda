#ifndef ALEXNET_CONVOLUTIONAL_CUH
#define ALEXNET_CONVOLUTIONAL_CUH
using namespace std;

#include "Layer.cuh"

class ConvLayer: public Layer{
public:
    ConvLayer(int input_size, int input_channel, int kernel_size, int kernel_channel, int stride, bool pad);
    ~ConvLayer();
    int getInputSize();
    int getInputChannel();
    int getKernelSize();
    int getKernelChannel();
    int getOutputSize();
    int getOutputChannel();
    float *forward(float *image) override;
    float *backpropagation(float* cost, float* back_neurons) override;
    void applyGradient(float lr) override;
private:
    int input_size;
    int input_channel;
    float *kernel;
    int kernel_size;
    int kernel_channel;
    int stride;
    int pad;
    int output_size;
    int output_channel;
};


#endif //ALEXNET_CONVOLUTIONAL_CUH
