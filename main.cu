#include <iostream>
#include <vector>
#include "matrixMul.cu"

using namespace std;


class fullLayer{
public:
	fullLayer(int num, fullLayer* next);
	int num_neurons;
	float* neurons;
	float** weigths;
	fullLayer* nextLayer;
	void forward();
};

int main() {
	vector<float> asd;
	fullLayer a = fullLayer(5, nullptr);
	fullLayer b = fullLayer(5, &a);
	b.forward();
	return 0;
}



fullLayer::fullLayer(const int num, fullLayer* next) {
	num_neurons = num;
	neurons = new float [num];
	nextLayer = next;
	if(next != nullptr){
		weigths = new float* [num];
		for (int i = 0; i < num; ++i)
			weigths[i] = new float [nextLayer->num_neurons];
	}
}

void fullLayer::forward() {
	nextLayer->neurons[0] = 3;
}
