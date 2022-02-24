#include "ConvLayer.cuh"
#include "CUDA/convolution.cuh"
#include "CUDA/vectorSum.cuh"


ConvLayer::ConvLayer(int input_size, int channels, int kernel_size, int kernel_num,
                     int stride, bool pad, Act func)
        : Layer(func){
    this->input_size = input_size;
    this->channels = channels;
    this->kernel_size = kernel_size;
    this->kernel_num = kernel_num;
    this->stride = stride;
    if(pad)
        this->pad = (kernel_size - 1) / 2;
    else
        this->pad = 0;
    this->output_size = ((input_size-kernel_size+2*pad)/stride+1);
    this->output_len = output_size*output_size*kernel_num;
    this->num_weights = kernel_size*kernel_size*channels*kernel_num;
    std::random_device generator;
	std::normal_distribution<float> weights_rand =
			std::normal_distribution<float>(0.f, sqrt(2.f/(float)(input_size*input_size*channels)));
    float * tmp_weights = new float[num_weights];
    for (int i=0; i<num_weights; i++){
        tmp_weights[i] = weights_rand(generator);
    }

    cudaMalloc(&this->weights,num_weights * sizeof(float));
    cudaMemcpy(this->weights, tmp_weights, num_weights * sizeof(float), cudaMemcpyHostToDevice);
	if(kernel_size == 7)
		save_CUDA(weights, num_weights, kernel_size, 0);
	delete[] tmp_weights;

	cudaMalloc(&this->bias,this->kernel_num * sizeof(float));
	cudaMalloc(&this->bias_derivative,this->kernel_num * sizeof(float));
    cudaMalloc(&this->weights_derivative,num_weights * sizeof(float));
    cudaMalloc(&this->current_weights_derivative,num_weights * sizeof(float));
    cudaMalloc(&this->prev_layer_derivative,input_size*input_size*channels * sizeof(float));
	cudaMalloc(&this->activation_derivative,output_len * sizeof(float));
    cudaMalloc(&this->activations,output_len * sizeof(float));
    cudaMemset(this->weights_derivative,0,num_weights * sizeof(float));
	cudaMemset(this->bias,0,kernel_num * sizeof(float));
    cudaMemset(this->bias_derivative,0,this->kernel_num * sizeof(float));
}

    ConvLayer::~ConvLayer(){
        Layer::~Layer();
        cudaFree(this->bias);
        cudaFree(this->bias_derivative);
        cudaFree(this->weights);
        cudaFree(this->activations);
        cudaFree(this->weights_derivative);

}

float* ConvLayer::forward(float *image) {
    convolution(image,
               this->weights,
               this->activations,
               this->input_size,
               this->kernel_size,
               this->stride,
               this->pad,
               this->channels,
               this->kernel_num);

//	vector_bias_sum(this->activations, bias, output_size*output_size, kernel_num);
    activation_func(this->activations, output_len);
    return this->activations;
}

int ConvLayer::getInputSize() {
    return this->input_size;
}
int ConvLayer::getChannel() {
    return this->channels;
}
int ConvLayer::getKernelSize() {
    return this->kernel_size;
}
int ConvLayer::getOutputSize() {
    return this->output_size;
}
int ConvLayer::getOutputChannel() {
    return this->kernel_num;
}

void ConvLayer::applyGradient(float lr) {
	vector_constant_mul(weights_derivative,lr,num_weights);
	vector_diff(weights,weights_derivative,num_weights);
	vector_constant_mul(bias_derivative,lr,kernel_num);
	vector_diff(bias,bias_derivative,kernel_num);
	cudaMemset(this->weights_derivative,0,num_weights * sizeof(float));
	cudaMemset(this->bias_derivative,0,kernel_num * sizeof(float));
	save_CUDA(weights, num_weights, kernel_size, kernel_size);
}

float *ConvLayer::backpropagation(float *cost, float *back_img) {
	if(kernel_num == 16)
		print_CUDA(cost, output_len);
	derivative_func(activations, activation_derivative, output_len);
	vector_mul(activation_derivative, cost, activation_derivative, output_len);
//	vector_conv_bias(bias_derivative, activation_derivative, output_size*output_size, kernel_num);
	cudaMemset(current_weights_derivative, 0, num_weights * sizeof(float));

	convolution_weights(back_img,
						activation_derivative,
						current_weights_derivative,
						this->input_size,
						this->output_size,
						this->stride,
						pad,
						channels,
						kernel_num);


	vector_sum(weights_derivative, current_weights_derivative, num_weights);

	convolution_prevlayer_backpropagation(activation_derivative,
			this->weights,
			prev_layer_derivative,
			this->output_size,
			this->kernel_size,
			kernel_size - 1,
			kernel_num,
			channels);

	return prev_layer_derivative;
}

int ConvLayer::getNeurons() {
    return output_len;
}

int ConvLayer::getNumBackNeurons() {
    return input_size*input_size*channels;
}
