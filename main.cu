#include <iostream>
#include "network.h"
#include <cmath>
#include <random>

using namespace std;

#define BATCH_SIZE 4
#define NUM_EPOCHS 300

int main() {
	random_device r;
	uniform_int_distribution<int> distribution = uniform_int_distribution<int>(0, 1);
    Network net(2, 1.0f);
    net.addFullLayer(4);
    net.addFullLayer(3);
    net.addFullLayer(1);
	float in[2][4] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
	float exp[4] = {0.0f, 1.0f, 1.0f, 1.0f};
	float* out;
	for (int j=0; j < NUM_EPOCHS; j++) {
		double loss = 0.0;
		for (int i = 0; i < BATCH_SIZE; i++) {
			//in[0] = (float)distribution(r);
			//in[1] = (float)distribution(r);
			float a[2] = {in[0][i] , in[1][i]};
			out = net.forward(a);
			net.train(out, &exp[i], a);

			loss += pow((out[0] - exp[i]), 2);
		}
		delete[] out;
		net.learn();
		cout <<"loss: " << loss / BATCH_SIZE << endl;
	}

    return 0;
}
