#ifndef ALEXNET_POOLING_CUH
#define ALEXNET_POOLING_CUH
using namespace std;

class PoolingLayer{
public:
    PoolingLayer(int input_size, int input_channel, int kernel_size, int stride, Act func);
    ~PoolingLayer();
    int getInputSize();
    int getInputChannel();
    int getPoolSize();
    int getOutputSize();
    int getOutputChannel();
    float *forward(float *image);
private:
    int input_size;
    int input_channel;
    int pool_size;
    int stride;
    int output_size;
    int output_channel;
};


#endif
