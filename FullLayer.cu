#include "FullLayer.cuh"

#if CUDA

FullLayer::FullLayer(int n_neurons, int linked_neurons, Act func) : Layer(func){
    /**
     * weights: number of rows = num_back_neurons, number of columns = num_neurons
     **/
    this->num_neurons = n_neurons;
    this->num_back_neurons = linked_neurons;
    this->num_weights = n_neurons*linked_neurons;
    std::random_device generator;
    std::uniform_real_distribution<float> weights_rand = std::uniform_real_distribution<float>(0.0f, 1.0f);
    float * tmp_weights = new float[num_weights];
    for (int i=0; i<num_weights; i++){
        //tmp_weights[i] = weights_rand(generator);
		tmp_weights[i] = 1.0f;
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
}

float* FullLayer::forward(float *values) {
    float *activationss = matrix_mul(values,
                           this->weights,
                           1,
                           this->getNumBackNeurons(),
                           this->getNeurons());

    vector_sum(activationss,bias,getNeurons());
    //cudaMemcpy(this->neurons, activationss, this->getNeurons()*sizeof(float), cudaMemcpyDeviceToDevice);
    activation_func(activationss, getNeurons());
    cudaMemcpy(this->activations, activationss, this->getNeurons()*sizeof(float), cudaMemcpyDeviceToDevice);

    return activationss;
}

float* FullLayer::backpropagation(float* cost, float* back_neurons) {
    // other derivatives are obtained in the same way as the bias derivative but using more terms
    // so we start computing bias derivatives and then use those as baseline for other derivatives


    float *der_fun = derivative_func(activations, getNeurons());
    float *current_bias_derivative = vector_mul(der_fun,cost,num_neurons);
    //cudaFree(der_fun);
    vector_sum(bias_derivative,current_bias_derivative,getNeurons());
    //cudaFree(cost);
    float* current_weights_derivative = matrix_mul(back_neurons,
                                                   current_bias_derivative,
                                                   this->getNumBackNeurons(),
                                                   1,
                                                   this->getNeurons());
    float* prev_layer_derivative = matrix_mul(this->weights,
                                              current_bias_derivative,
                                              this->getNumBackNeurons(),
                                              this->getNeurons(),
                                              1);

    //cudaFree(current_bias_derivative);

    vector_sum(weights_derivative,current_weights_derivative,num_weights);


    //cudaFree(current_weights_derivative);
    float *x;
    cudaMalloc(&x, getNumBackNeurons() * sizeof(float));
    cudaMemcpy(x, prev_layer_derivative, getNumBackNeurons() * sizeof(float), cudaMemcpyDeviceToDevice);

    return x;
}

void FullLayer::applyGradient(float lr) {
//    float *ee = new float[num_weights];
//    cudaMemcpy(ee,weights,num_weights*sizeof (float),cudaMemcpyDeviceToHost);
//    printf("\n weights pre diff:\n");
//    for(int i=0;i<this->num_weights;i++)
//        printf("%f ",ee[i]);
//    float *ff = new float[this->getNeurons()];
//    cudaMemcpy(ff,bias,this->getNeurons()*sizeof (float),cudaMemcpyDeviceToHost);
//    printf("\n bias pre diff:\n");
//    for(int i=0;i<this->getNeurons();i++)
//        printf("%f ",ff[i]);
//    float *cc = new float[num_weights];
//    cudaMemcpy(cc,weights_derivative,num_weights*sizeof (float),cudaMemcpyDeviceToHost);
//    printf("\n weights derivative:\n");
//    for(int i=0;i<this->num_weights;i++)
//        printf("%f ",cc[i]);
//    float *dd = new float[num_neurons];
//    cudaMemcpy(dd,bias_derivative,num_neurons*sizeof (float),cudaMemcpyDeviceToHost);
//    printf("\n bias derivative:\n");
//    for(int i=0;i<this->num_neurons;i++)
//        printf("%f ",dd[i]);

    vector_constant_mul(weights_derivative,lr,num_weights);
    vector_diff(weights,weights_derivative,num_weights);
    vector_constant_mul(bias_derivative,lr,num_neurons);
    vector_diff(bias,bias_derivative,num_neurons);

//
//    float *aa = new float[num_weights];
//    cudaMemcpy(aa,weights,num_weights*sizeof (float),cudaMemcpyDeviceToHost);
//    printf("\n weights post diff:\n");
//    for(int i=0;i<this->num_weights;i++)
//        printf("%f ",aa[i]);
//
//    float *bb = new float[this->getNeurons()];
//    cudaMemcpy(bb,bias,this->getNeurons()*sizeof (float),cudaMemcpyDeviceToHost);
//    printf("\n bias post diff:\n");
//    for(int i=0;i<this->getNeurons();i++)
//        printf("%f ",bb[i]);
//
//
//    printf("\n\n");
    cudaMemset(this->weights_derivative,0,num_weights * sizeof(float));
    cudaMemset(this->bias_derivative,0,num_neurons * sizeof(float));
}

int FullLayer::getNeurons() {
    return num_neurons;
}

int FullLayer::getNumBackNeurons() {
    return num_back_neurons;
}


#else

FullLayer::FullLayer(int n_neurons, int linked_neurons, Act func) : Layer(func){
    /**
     * weights: number of rows = weights_len, number of columns = num_neurons
     **/
	this->num_neurons = n_neurons;
	this->num_back_neurons = linked_neurons;
	this->num_weights = n_neurons*linked_neurons;
	this->weights = new float[num_weights];
	this->weights_derivative = new float[num_weights]();
	this->activations = new float[n_neurons];
	this->bias = new float[n_neurons]();
	this->bias_derivative = new float[n_neurons]();
	std::random_device generator;
	std::uniform_real_distribution<float> weights_rand = std::uniform_real_distribution<float>(0.0f, 1.0f);
	for (int i=0; i<num_weights; i++){
		weights[i] = weights_rand(generator);
//		weights[i] = 1.0f;
	}
}

FullLayer::~FullLayer(){
	Layer::~Layer();
	delete[] this->bias;
    delete[] this->bias_derivative;
	delete[] this->weights;
	delete[] this->activations;
	delete[] this->weights_derivative;
}

float* FullLayer::forward(float *values) {
	float *val =matrix_mul(values,
                           this->weights,
                           1,
                           this->getNumBackNeurons(),
                           this->getNeurons());

	for(int i=0; i<getNeurons(); i++){
		val[i] += bias[i];
		val[i] = activation_func(val[i]);
        this->activations[i] = val[i];
	}
	return val;
}

float* FullLayer::backpropagation(float* cost, float* back_neurons) {
    // other derivatives are obtained in the same way as the bias derivative but using more terms
    // so we start computing bias derivatives and then use those as baseline for other derivatives
	float* current_bias_derivative = new float[num_neurons];
    for(int i = 0; i < num_neurons; i++){
        current_bias_derivative[i] = derivative_func(this->activations[i]) * cost[i];
		bias_derivative[i] += current_bias_derivative[i];
    }

	delete[] cost;
	float* current_weights_derivative = matrix_mul_CPU(back_neurons,
                                                       current_bias_derivative,
                                                       this->num_back_neurons,
                                                       1,
                                                       this->num_neurons);
    float* prev_layer_derivative = matrix_mul_CPU(this->weights,
                                                  current_bias_derivative,
                                                  this->getNumBackNeurons(),
                                                  this->getNeurons(),
                                                  1);

	delete[] current_bias_derivative;

	for (int i=0; i<num_weights; i++){
		weights_derivative[i] += current_weights_derivative[i];
	}
//    vector_sum(weights_derivative, current_weights_derivative, num_weights);
//    weights_derivative_CPU = vector_sum_CPU(weights_derivative, current_weights_derivative, num_weights);

	delete[] current_weights_derivative;
	return prev_layer_derivative;
}

void FullLayer::applyGradient(float lr) {
    //fare in CUDA
	for (int i=0; i<num_weights; i++){
		weights[i] -= weights_derivative[i] * lr;
		weights_derivative[i] = 0;
	}
	for(int i = 0; i < this->getNeurons(); i++){
		bias[i] -= bias_derivative[i] * lr;
		bias_derivative[i] = 0;
	}
}

int FullLayer::getNeurons() {
	return num_neurons;
}

int FullLayer::getNumBackNeurons() {
	return num_back_neurons;
}


#endif

