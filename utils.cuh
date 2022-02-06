#ifndef ALEXNET_UTLIS_H
#define ALEXNET_UTLIS_H

#endif //ALEXNET_UTLIS_H

static float reLU(float f){
	return f > 0.0f ? f : 0.0f;
}

static float Heaviside(float f){
	return f > 0.0f ? 1.0f : 0.0f;
}

static float sigmoid(float f){
	return 1.f/ (1.f + exp(-f));
}

static float der_sigmoid(float f) {
	return f * (1 - f);
}

static double sech2(float f) {
	float sh = 1.0f / std::cosh(f);   // sech(x) == 1/cosh(x)
	return sh*sh;                     // sech^2(x)
}

float* Softmax(float input[], int length){
	float sum = 0;
	for(int i = 0; i < length; i++) {
		input[i] = exp(input[i]);
		sum += input[i];
	}
	for(int i = 0; i < length; i++)
		input[i] = input[i]/sum;
	return input;
}