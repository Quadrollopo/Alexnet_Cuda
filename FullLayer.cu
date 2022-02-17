#include "FullLayer.cuh"

FullLayer::FullLayer(int n_neurons, int linked_neurons, Act func) : Layer(func){
    /**
     * weights: number of rows = num_back_neurons, number of columns = num_neurons
     **/
    this->num_neurons = n_neurons;
    this->num_back_neurons = linked_neurons;
    this->num_weights = n_neurons*linked_neurons;
    std::random_device generator;
    std::uniform_real_distribution<float> weights_rand = std::uniform_real_distribution<float>(0.0f, 0.1f);
    float * tmp_weights = new float[num_weights];
    for (int i=0; i<num_weights; i++){
        tmp_weights[i] = weights_rand(generator);
    }
    cudaMalloc(&this->weights,num_weights * sizeof(float));
    cudaMemcpy(this->weights, tmp_weights, num_weights * sizeof(float), cudaMemcpyHostToDevice);
    delete[] tmp_weights;
    float *tmp_weights_der=new float[num_weights]();
    float *tmp_bias=new float[n_neurons]();
    float *tmp_bias_der=new float[n_neurons]();
    cudaMalloc(&this->weights_derivative,num_weights * sizeof(float));
    cudaMemset(this->weights_derivative,0,num_weights * sizeof(float));
    cudaMalloc(&this->activations,n_neurons * sizeof(float));
    cudaMalloc(&this->bias,n_neurons * sizeof(float));
    cudaMemset(this->bias,0,n_neurons * sizeof(float));
    cudaMalloc(&this->bias_derivative,n_neurons * sizeof(float));
    cudaMemset(this->bias_derivative,0,n_neurons * sizeof(float));
    cudaMalloc(&this->neurons,n_neurons * sizeof(float));
    cudaMalloc(&this->current_weights_derivative,num_weights * sizeof(float));
    cudaMalloc(&this->prev_layer_derivative,num_back_neurons * sizeof(float));
    cudaMalloc(&this->activation_derivative,num_neurons * sizeof(float));
    cudaMalloc(&this->current_bias_derivative,num_neurons * sizeof(float));

    delete[] tmp_weights_der;
    delete[] tmp_bias_der;
    delete[] tmp_bias;
}

FullLayer::~FullLayer(){
    Layer::~Layer();
    cudaFree(this->bias);
    cudaFree(this->bias_derivative);
    cudaFree(this->weights);
    cudaFree(this->activations);
    cudaFree(this->neurons);
    cudaFree(this->weights_derivative);
    cudaFree(this->current_weights_derivative);
    cudaFree(this->prev_layer_derivative);
    cudaFree(this->activation_derivative);
    cudaFree(this->current_bias_derivative);
}

float* FullLayer::forward(float *values) {
    matrix_mul(values,
               this->weights,
               activations,
               1,
               this->getNumBackNeurons(),
               this->getNeurons());
    vector_sum(activations,bias,getNeurons());
    activation_func(activations, num_neurons);
    return activations;
}

float* FullLayer::backpropagation(float* cost, float* back_neurons) {
    // other derivatives are obtained in the same way as the bias derivative but using more terms
    // so we start computing bias derivatives and then use those as baseline for other derivatives


    derivative_func(activations, activation_derivative, getNeurons());
    vector_mul(activation_derivative,cost,current_bias_derivative,num_neurons);
    vector_sum(bias_derivative,current_bias_derivative,getNeurons());
    matrix_mul3(back_neurons,
               current_bias_derivative,
               current_weights_derivative,
               this->getNumBackNeurons(),
               1,
               this->getNeurons());
    matrix_mul3(this->weights,
               current_bias_derivative,
               prev_layer_derivative,
               this->getNumBackNeurons(),
               this->getNeurons(),
               1);
    vector_sum(this->weights_derivative,current_weights_derivative,num_weights);
    return prev_layer_derivative;
}

void FullLayer::applyGradient(float lr) {

    vector_constant_mul(this->weights_derivative,lr,num_weights);
    vector_diff(weights,weights_derivative,num_weights);
    vector_constant_mul(bias_derivative,lr,num_neurons);
    vector_diff(bias,bias_derivative,num_neurons);
    cudaMemset(this->weights_derivative,0,num_weights * sizeof(float));
    cudaMemset(this->bias_derivative,0,num_neurons * sizeof(float));
}

int FullLayer::getNeurons() {
    return num_neurons;
}

int FullLayer::getNumBackNeurons() {
    return num_back_neurons;
}

