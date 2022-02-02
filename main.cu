#include <iostream>
#include <vector>

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
//	fullLayer* layer = new fullLayer(5, nullptr);
//	for (int i = 0; i < 4; i++){
//		layer = new fullLayer((i + 2)*2, layer);
//	}
	std::vector<float*> x;
	float y[10];
	int i, j;
	for (i=0; i<10; i++)
		for (j=0; j<10; j++)
			y[j] = (float) i*10 + j;
		x.push_back(y);
	for (i=0; i<10; i++)
		for (j=0; j<10; j++)
			std::cout<<x[i][j]<<std::endl;


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
