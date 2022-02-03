#include <iostream>
#include <vector>
#include "network.h"
#include "matrixMul.cuh"
using namespace std;


int main() {

	Network net(5);
	net.addFullLayer(4);
	net.addFullLayer(3);
	float in[] = {3.0f, 2.0f, 2.5f, 2.0f, 2.0f};
	float* out = net.forward(in);
	for (int i = 0; i < 3; i++)
		cout << out[i] << endl;
	matrixMul<<<2,2>>>();
	return 0;
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
