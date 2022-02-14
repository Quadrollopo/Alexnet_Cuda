#ifndef ALEXNET_CONVOLUTIONAL_CUH
#define ALEXNET_CONVOLUTIONAL_CUH
using namespace std;

#include "Layer.cuh"
#include <memory>
#include <random>
#include <cuda_runtime.h>
#include "CUDA_or_CPU.cuh"
class ConvLayer: public Layer{
public:
    ConvLayer(int input_size, int channels, int kernel_size, int kernel_num, int stride, bool pad, Act func);
    ~ConvLayer();
    int getInputSize();
    int getKernelSize();
	int getChannel();
	int getOutputSize();
    int getOutputChannel();
    float *forward(float *image) override;
    float *backpropagation(float* cost, float* back_neurons) override;
    void applyGradient(float lr) override;
	int getNeurons() override;
	int getNumBackNeurons() override;
private:
    int input_size;	//lato dell'immagine
    int channels;	//profondità
    int kernel_size;	//lato del kernel
    int kernel_num;	//numero di kernel
    int stride;
    int pad;
    int output_size;	// lato dell'output
	int output_len; //output_size*output_size*kernel_num
};


#endif //ALEXNET_CONVOLUTIONAL_CUH
