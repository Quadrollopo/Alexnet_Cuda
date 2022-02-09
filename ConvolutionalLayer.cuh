#ifndef ALEXNET_CONVOLUTIONAL_CUH
#define ALEXNET_CONVOLUTIONAL_CUH
using namespace std;

class ConvolutionalLayer{
public:
    ConvolutionalLayer(int input_size, int input_channel, int kernel_size, int kernel_channel, int stride, int pad);
    ~ConvolutionalLayer();
    int getInputSize();
    int getInputChannel();
    int getKernelSize();
    int getKernelChannel();
    int getOutputSize();
    int getOutputChannel();
    float *forward(float *image);
    float *backpropagation();
    void applyGradient();
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
